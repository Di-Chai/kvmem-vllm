"""Opt-in, process-local native v2 InputBatch diagnostic; no CUDA operations.

Fresh diagnostic profiles may set worker_extension_cls to
native_executed_batch_observer.Extension and NATIVE_BATCH_OBSERVER_CONFIG to a
frozen JSON config. Importing this module without that variable is a no-op.
Successful target/sampler completion is not scheduler/client token commitment.
"""
from dataclasses import asdict, dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import weakref


ATTR = '_native_executed_batch_observer'
TICKET = '_native_executed_batch_ticket'
SOURCE_FILES = (
    'v1/worker/worker_base.py', 'v1/worker/gpu_worker.py',
    'v1/worker/gpu/model_runner.py', 'v1/worker/gpu/async_utils.py',
    'v1/worker/gpu/states.py', 'v1/worker/gpu/input_batch.py',
)


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class Batch:
    execution_id: int
    dispatch_wall: float
    dispatch_monotonic: float
    req_ids: tuple
    slots: tuple
    slot_generations: tuple
    is_prefilling: tuple
    prompt_lengths: tuple
    prefill_lengths: tuple
    computed_prefill_lengths: tuple
    scheduled_token_upper_counts: tuple
    draft_token_upper_counts: tuple
    num_reqs: int
    padded_reqs: int
    num_tokens: int
    padded_tokens: int
    full_graph: bool


class Writer:
    """Bounded queue and output; serialization and disk I/O stay off execution threads."""
    def __init__(self, directory, *, capacity=128, max_records=20000,
                 max_bytes=64*1024**2, max_record_bytes=32768):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.queue = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.max_records,self.max_bytes,self.max_record_bytes = max_records,max_bytes,max_record_bytes
        self.counts = dict(written=0,bytes=0,dropped=0,io_errors=0)
        self.thread = threading.Thread(target=self._run,name='native-batch-log',daemon=True)
        self.thread.start()

    def emit(self, record):
        if self.stop.is_set():
            with self.lock:self.counts['dropped'] += 1
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with self.lock:self.counts['dropped'] += 1

    def _run(self):
        try:
            with (self.directory/'events.jsonl').open('x') as stream:
                while not self.stop.is_set() or not self.queue.empty():
                    try:record = self.queue.get(timeout=.05)
                    except queue.Empty:continue
                    try:
                        value = dict(record)
                        if isinstance(value.get('batch'),Batch):value['batch'] = asdict(value['batch'])
                        line = json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
                        size = len(line.encode())
                        with self.lock:
                            keep = (self.counts['written'] < self.max_records and
                                    self.counts['bytes']+size <= self.max_bytes and size <= self.max_record_bytes)
                            if not keep:self.counts['dropped'] += 1
                        if keep:
                            stream.write(line)
                            with self.lock:
                                self.counts['written'] += 1
                                self.counts['bytes'] += size
                    except Exception:
                        with self.lock:self.counts['io_errors'] += 1
                    finally:self.queue.task_done()
        except Exception:
            with self.lock:self.counts['io_errors'] += 1

    def snapshot(self):
        with self.lock:result = dict(self.counts)
        return dict(**result,queued=self.queue.qsize(),writer_alive=self.thread.is_alive())

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        return self.snapshot()


class Observer:
    def __init__(self, runner, writer, *, max_pending=8, max_requests=32):
        self.runner = weakref.ref(runner)
        self.writer = writer
        self.max_pending,self.max_requests = max_pending,max_requests
        self.lock = threading.Lock()
        self.local = threading.local()
        self.pending = {}
        self.pending_sample = None  # only an integer state identity and a frozen CPU Batch
        self.generations = {}
        self.sequence = 0
        self.counts = dict(dispatches=0,completed=0,failed=0,unmatched=0,dropped=0,observer_errors=0)
        self.counts['post_close_dispatches'] = 0
        self.armed = True
        self.explicit_seal = False

    def error(self):
        with self.lock:self.counts['observer_errors'] += 1

    def allocation(self, state, req_id):
        slot = int(state.req_id_to_index[req_id])
        self.generations[slot] = self.generations.get(slot,0)+1

    def begin(self, eligible):
        frame = SimpleNamespace(eligible=eligible,batch=None,
                                input_batch_identity=None,previous=getattr(self.local,'frame',None))
        self.local.frame = frame
        if frame.previous is not None:
            frame.eligible = False
            self.error()
        if self.pending_sample is not None:
            _,old = self.pending_sample
            self.pending_sample = None
            self.fail(old,'sample_boundary_missing')
        return frame

    def end(self, frame):
        self.local.frame = frame.previous

    def capture(self, input_batch, full_graph):
        frame = getattr(self.local,'frame',None)
        if frame is None or not frame.eligible:return
        with self.lock:
            if not self.armed:
                # Native work is untouched, but it invalidates a prior seal.
                self.counts['post_close_dispatches'] += 1
                return
        require(frame.batch is None,'More than one target dispatch in an execute call')
        runner = self.runner()
        count = int(input_batch.num_reqs)
        require(0 < count <= self.max_requests,'Diagnostic request bound exceeded')
        ids = tuple(input_batch.req_ids)
        require(len(ids) == count and len(set(ids)) == count and
                all(isinstance(x,str) and len(x) <= 256 for x in ids),'Invalid or oversized request IDs')
        def values(name,convert=int):
            result = tuple(convert(x) for x in getattr(input_batch,name))
            require(len(result) == count,'Batch field has wrong length: '+name)
            return result
        slots = values('idx_mapping_np')
        require(all(runner.req_states.req_id_to_index[r] == s for r,s in zip(ids,slots)),
                'InputBatch owner slot mismatch')
        phase = values('is_prefilling_np',bool)
        prefill,computed = values('prefill_len_np'),values('num_computed_prefill_tokens_np')
        require(phase == tuple(a < b for a,b in zip(computed,prefill)),'InputBatch phase mismatch')
        scheduled = values('num_scheduled_tokens')
        require(all(x > 0 for x in scheduled),'Zero-row request in actual dispatch')
        drafts = ((0,)*count if input_batch.num_draft_tokens_per_req is None
                  else values('num_draft_tokens_per_req'))
        with self.lock:
            self.sequence += 1
            if len(self.pending) >= self.max_pending:
                self.counts['dropped'] += 1
                return
            batch = Batch(self.sequence,time.time(),time.monotonic(),ids,slots,
                tuple(self.generations.get(s,0) for s in slots),phase,
                tuple(int(runner.req_states.prompt_len.np[s]) for s in slots),prefill,computed,scheduled,drafts,
                count,int(input_batch.num_reqs_after_padding),int(input_batch.num_tokens),
                int(input_batch.num_tokens_after_padding),bool(full_graph))
            self.pending[batch.execution_id] = batch
            self.counts['dispatches'] += 1
        frame.batch = batch
        frame.input_batch_identity = id(input_batch)
        self.writer.emit(dict(kind='dispatch',batch=batch,executed=False,
            length_scope='prompt_only_lower_bound_for_decode; scheduled counts are CPU upper bounds'))

    def executed(self, frame, state, returned):
        if frame.batch is None:return
        if (returned is not None or state is None or id(state.input_batch) != frame.input_batch_identity or
                tuple(state.input_batch.req_ids) != frame.batch.req_ids):
            self.fail(frame.batch,'execute_state_mismatch')
            return
        self.pending_sample = (id(state),frame.batch)

    def take_sample(self, state):
        pending,self.pending_sample = self.pending_sample,None
        if pending is None:return None
        identity,batch = pending
        if state is None or id(state) != identity:
            self.fail(batch,'sample_state_mismatch')
            return None
        return batch

    def fail(self, batch, reason):
        if batch is None:return
        with self.lock:
            if self.pending.pop(batch.execution_id,None) is None:
                self.counts['unmatched'] += 1
                return
            self.counts['failed'] += 1
            self.writer.emit(dict(kind='failed',execution_id=batch.execution_id,executed=False,reason=reason))

    def complete(self, batch, result):
        ids = tuple(result.req_ids)
        if ids != batch.req_ids or len(result.sampled_token_ids) != batch.num_reqs:
            self.fail(batch,'output_owner_mismatch')
            return
        counts = tuple(len(x) for x in result.sampled_token_ids)
        with self.lock:
            if self.pending.pop(batch.execution_id,None) is None:
                self.counts['unmatched'] += 1
                return
            self.counts['completed'] += 1
            # Enqueue before making the no-pending state visible to seal().
            self.writer.emit(dict(kind='completed',execution_id=batch.execution_id,executed=True,
                completed_wall=time.time(),completed_monotonic=time.monotonic(),raw_sampled_counts=counts,
                scope='Successful target/sampler AsyncOutput; not final client commitment or MTP proposal completion'))

    def snapshot(self):
        with self.lock:result = dict(self.counts,pending=len(self.pending),
            pending_sample=self.pending_sample is not None,armed=self.armed,explicit_seal=self.explicit_seal)
        result.update(writer=self.writer.snapshot(),max_pending=self.max_pending)
        result['complete_coverage'] = (not result['armed'] and result['dispatches'] > 0 and
            result['dispatches'] == result['completed'] and not result['pending_sample'] and
            result['pending'] == result['failed'] == result['unmatched'] ==
            result['dropped'] == result['observer_errors'] == result['post_close_dispatches'] ==
            result['writer']['dropped'] == result['writer']['io_errors'] == result['writer']['queued'] == 0 and
            not result['writer']['writer_alive'])
        return result

    def close(self):
        with self.lock:self.armed = False
        self.writer.close()
        return self.snapshot()

    def seal(self):
        """CPU-only final flush, after the controller has stopped and drained work."""
        runner = self.runner()
        with self.lock:
            require(not self.pending and self.pending_sample is None,
                    'Cannot seal with pending execution or sampling')
            require(runner is not None and runner.execute_model_state is None and
                    getattr(self.local,'frame',None) is None,
                    'Cannot seal during a target execution')
            self.armed = False
            self.explicit_seal = True
        report = self.close()
        report['scope'] = ('Coverage through this seal RPC only; any later target dispatch invalidates '
            'coverage. Final snapshot/summary must be checked if any work follows. '
            'CPU metadata only; not final client token commitment.')
        temporary = self.writer.directory/'summary.json.tmp'
        try:
            temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(self.writer.directory/'summary.json')
        except Exception:
            self.error()
            raise
        return report


def safely(observer, function, *args):
    try:return function(*args)
    except Exception:
        observer.error()
        return None


def validate_runner(runner):
    p = runner.parallel_config
    require(all(getattr(p,k) == 1 for k in ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')),
            'Only single-GPU TP/PP/DP=1 is qualified for this adapter')
    require(runner.dcp_size == 1 and runner.pcp_manager is None and runner.ubatch_runner is None,
            'CP/microbatch runner is unsupported')
    require(not runner.is_pooling_model and runner.lora_config is None,
            'Pooling/LoRA is outside this diagnostic scope')
    require(runner.adaptive_verification is None,'Adaptive compaction needs additional device metadata')
    require(runner.vllm_config.kv_transfer_config is None,'This observer targets native execution without a connector')


def install(classes, config, writer_factory=Writer):
    """Install wrappers in this process; injected fake classes support CPU tests."""
    require(config.get('enabled') is True,'Observer must be explicitly enabled')
    changed = []
    def patch(cls,name,builder):
        original = getattr(cls,name)
        require(not getattr(original,'_native_batch_observer_wrapper',False),'Observer already installed')
        wrapper = functools.wraps(original)(builder(original))
        wrapper._native_batch_observer_wrapper = True
        setattr(cls,name,wrapper);changed.append((cls,name,original))
    def compile_wrapper(original):
        def call(worker,*args,**kwargs):
            require(isinstance(worker.model_runner,classes.Runner),'Only the v2 model runner is supported')
            result = original(worker,*args,**kwargs)
            runner = worker.model_runner
            validate_runner(runner)
            require(getattr(runner,ATTR,None) is None,'Runner already armed')
            writer = writer_factory(Path(config['output_directory'])/f'worker-{os.getpid()}')
            observer = Observer(runner,writer)
            for obj in (runner,runner.step_timing,runner.req_states):setattr(obj,ATTR,observer)
            return result
        return call
    def execute_wrapper(original):
        def call(runner,scheduler_output,intermediate_tensors=None,dummy_run=False,
                 skip_attn_for_dummy_run=False,is_profile=False,context_len=0,valid_dummy_state_slots=False):
            observer = getattr(runner,ATTR,None)
            frame = safely(observer,observer.begin,not dummy_run and not is_profile) if observer else None
            try:
                result = original(runner,scheduler_output,intermediate_tensors,dummy_run,
                    skip_attn_for_dummy_run,is_profile,context_len,valid_dummy_state_slots)
                if frame is not None:safely(observer,observer.executed,frame,runner.execute_model_state,result)
                return result
            except BaseException:
                if frame is not None:safely(observer,observer.fail,frame.batch,'execute_exception')
                raise
            finally:
                if frame is not None:safely(observer,observer.end,frame)
        return call
    def record_wrapper(original):
        def call(collector,input_batch,full_cudagraph):
            result = original(collector,input_batch,full_cudagraph)
            observer = getattr(collector,ATTR,None)
            if observer:safely(observer,observer.capture,input_batch,full_cudagraph)
            return result
        return call
    def sample_wrapper(original):
        def call(runner,*args,**kwargs):
            observer = getattr(runner,ATTR,None)
            batch = safely(observer,observer.take_sample,runner.execute_model_state) if observer else None
            try:
                result = original(runner,*args,**kwargs)
            except BaseException:
                if batch:safely(observer,observer.fail,batch,'sample_exception')
                raise
            if batch:
                if isinstance(result,classes.AsyncOutput):
                    # No bound get_output closure, no InputBatch/GPU tensor reference.
                    setattr(result,TICKET,(observer,batch))
                else:safely(observer,observer.fail,batch,'missing_async_output')
            return result
        return call
    def output_wrapper(original):
        def call(output,*args,**kwargs):
            # Clearing first makes repeated native calls observable at most once.
            ticket = getattr(output,TICKET,None)
            if ticket is not None:
                with ticket[0].lock:
                    ticket = getattr(output,TICKET,None)
                    if ticket is not None:delattr(output,TICKET)
            try:result = original(output,*args,**kwargs)
            except BaseException:
                if ticket:safely(ticket[0],ticket[0].fail,ticket[1],'output_exception')
                raise
            if ticket:safely(ticket[0],ticket[0].complete,ticket[1],result)
            return result
        return call
    def add_wrapper(original):
        def call(state,req_id,*args,**kwargs):
            result = original(state,req_id,*args,**kwargs)
            observer = getattr(state,ATTR,None)
            if observer:safely(observer,observer.allocation,state,req_id)
            return result
        return call
    def shutdown_wrapper(original):
        def call(worker,*args,**kwargs):
            observer = getattr(worker.model_runner,ATTR,None)
            try:return original(worker,*args,**kwargs)
            finally:
                if observer:
                    summary = safely(observer,observer.close)
                    if summary is not None:
                        try:(observer.writer.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                        except Exception:observer.error()
        return call
    try:
        patch(classes.Worker,'compile_or_warm_up_model',compile_wrapper)
        patch(classes.Runner,'execute_model',execute_wrapper)
        patch(classes.Timing,'record_batch',record_wrapper)
        patch(classes.Runner,'sample_tokens',sample_wrapper)
        patch(classes.AsyncOutput,'get_output',output_wrapper)
        patch(classes.RequestState,'add_request',add_wrapper)
        patch(classes.Worker,'shutdown',shutdown_wrapper)
    except BaseException:
        for cls,name,original in reversed(changed):setattr(cls,name,original)
        raise
    def undo():
        for cls,name,original in reversed(changed):setattr(cls,name,original)
    return undo


def install_from_file(path):
    config = json.loads(Path(path).read_text())
    require(config.get('schema') == 'native_batch_observer_v1','Unknown diagnostic config')
    require(set(config.get('source_sha256',{})) == set(SOURCE_FILES),'Exact native hook source identities required')
    import vllm
    source = Path(vllm.__file__).resolve().parent
    for name,expected in config['source_sha256'].items():
        require(hashlib.sha256((source/name).read_bytes()).hexdigest() == expected,'Native hook source changed: '+name)
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.async_utils import AsyncOutput,StepTimingCollector
    from vllm.v1.worker.gpu.states import RequestState
    return install(SimpleNamespace(Worker=Worker,Runner=GPUModelRunner,AsyncOutput=AsyncOutput,
        Timing=StepTimingCollector,RequestState=RequestState),config)


class Extension:
    """Unique optional RPC only; never override an existing worker method."""
    def native_executed_batch_snapshot(self):
        observer = getattr(self.model_runner,ATTR,None)
        return None if observer is None else observer.snapshot()

    def native_executed_batch_seal(self):
        observer = getattr(self.model_runner,ATTR,None)
        require(observer is not None,'Native batch observer is not armed')
        return observer.seal()


if os.environ.get('NATIVE_BATCH_OBSERVER_CONFIG'):
    install_from_file(os.environ['NATIVE_BATCH_OBSERVER_CONFIG'])
