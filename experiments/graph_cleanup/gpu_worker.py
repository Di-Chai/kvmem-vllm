"""Eager single-request shadow KV exclusion on the real native paged cache.

Historical KV bytes are retained for audit, but masked positions cannot contribute
to probe/decode attention. RoPE positions, all prompt IDs, recurrent history and
non-candidate KV stay unchanged. This is not physical storage reclamation.
"""
import hashlib
import math
import time

import torch


def sha(tensor):
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


class Measurement:
    def __init__(self, runner):
        self.runner = runner
        self.active = None
        self.reference = {}
        self.reference_history = {}
        self.workspace = None
        self.wrapper = None
        self.plan_key = None
        self.positions = None
        self.modules = dict(runner.model.named_modules())
        self.handles = []
        self.attention_names = []
        self.layer_names = []
        self.handles.append(runner.model.register_forward_pre_hook(self.before_model, with_kwargs=True))
        for name, module in self.modules.items():
            if name.endswith('.self_attn.attn'):
                self.attention_names.append(name)
                original = module.impl.forward
                module.impl.forward = self.wrap_attention(name, original)
            if '.model.layers.' in name and name.rsplit('.',1)[1].isdigit():
                self.layer_names.append(name)
                self.handles.append(module.register_forward_hook(self.layer_hook(name)))
        if len(self.attention_names)!=16 or len(self.layer_names)!=64:
            raise RuntimeError(f'Unexpected model geometry: {len(self.attention_names)}, {len(self.layer_names)}')

    def configure(self, config):
        if config.get('reference'):
            self.reference = {}
            self.reference_history = {}
        self.active = dict(config)
        self.slots = {}
        self.page_tables = {}
        self.metrics = {}
        self.history_hashes = {}
        self.cache_geometry = {}
        self.attention_mass = {}
        self.native_relative = []
        self.events = 0
        self.plan_key = None
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        return dict(attention_layers=self.attention_names, transformer_layers=self.layer_names)

    def before_model(self, module, args, kwargs):
        if self.active is None:
            return
        positions = kwargs.get('positions', args[1] if len(args)>1 else None)
        if positions is None:
            raise RuntimeError('Missing native absolute positions')
        if positions.ndim==2:
            if not torch.equal(positions[0],positions[1]) or not torch.equal(positions[0],positions[2]):
                raise RuntimeError('Non-text MRoPE input')
            positions=positions[0]
        self.positions = positions.detach().cpu().tolist()
        if self.positions != list(range(self.positions[0], self.positions[0]+len(self.positions))):
            raise RuntimeError('This diagnostic requires a single contiguous request')

    def record(self, kind, name, tensor):
        config=self.active
        if config is None or self.positions is None:
            return
        rows=[i for i,p in enumerate(self.positions[:tensor.shape[0]])
              if config['query_start'] <= p < config['query_end']]
        if not rows:
            return
        pos=[self.positions[i] for i in rows]
        data=tensor[rows].detach().cpu().clone()
        key=kind+'/'+name
        if config.get('reference'):
            ref=self.reference.setdefault(key,{})
            for p,t in zip(pos,data): ref[p]=t.clone()
            self.metrics.setdefault(key,[]).extend(dict(position=p, norm=float(t.float().norm())) for p,t in zip(pos,data))
        else:
            ref=torch.stack([self.reference[key][p] for p in pos]).float()
            cur=data.float()
            diff=(cur-ref).reshape(len(pos),-1).norm(dim=1)
            norms=ref.reshape(len(pos),-1).norm(dim=1)
            rel=diff/(norms+1e-8)
            self.metrics.setdefault(key,[]).extend(dict(position=p, relative_l2=float(d),
                reference_norm=float(n), difference_norm=float(z)) for p,d,n,z in zip(pos,rel,norms,diff))

    def layer_hook(self, name):
        def hook(module,args,result):
            if (self.active is None or self.positions[-1]<self.active['query_start']
                    or self.positions[0]>=self.active['query_end']):
                return
            hidden,residual=result[:2]
            # vLLM delays the residual add until the next layer's fused norm.
            # The mathematical Transformer output is hidden + residual.
            total=hidden.float()+residual.float()
            self.record('transformer_residual_sum',name,total)
        return hook

    def wrap_attention(self,name,original):
        def forward(layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs):
            config=self.active
            # KV update is a separate native op, already performed by
            # Attention.forward. After the fixed probe, the original attention
            # result would be discarded: avoid computing it a second time.
            skip_original=(config is not None and config['mode']!='native'
                           and self.positions[0]>=config['query_end'])
            if skip_original:
                if layer.attn_backend.forward_includes_kv_cache_update:
                    raise RuntimeError('Cannot bypass a fused KV-write backend')
                result=output
            else:
                result=original(layer,query,key,value,kv_cache,attn_metadata,output,*args,**kwargs)
            if config is None or attn_metadata is None:
                return result
            count=int(attn_metadata.num_actual_tokens)
            pos=self.positions[:count]
            slots=attn_metadata.slot_mapping[:count].detach().cpu().tolist()
            logical=self.slots.setdefault(name,{})
            logical.update(zip(pos,slots))
            if pos[-1]<config['query_start']:
                return result
            if kv_cache.ndim!=4 or kv_cache.shape[1]!=4 or kv_cache.shape[-1]!=512:
                raise RuntimeError(f'Unsupported native KV geometry/layout: {tuple(kv_cache.shape)}, {attn_metadata.kv_cache_layout}')
            page=int(kv_cache.shape[2]);length=pos[-1]+1
            if len(logical)!=length or min(slots)<0:
                raise RuntimeError('Incomplete actual KV mapping (prefix reuse or preemption is not supported)')
            if name not in self.page_tables:
                all_slots=[logical[p] for p in range(length)]
                if any(s%page!=p%page for p,s in enumerate(all_slots)):
                    raise RuntimeError('Unexpected native slot offsets')
                pages=[all_slots[p]//page for p in range(0,length,page)]
                if any(all_slots[p]//page!=pages[p//page] for p in range(length)):
                    raise RuntimeError('Discontinuous physical page mapping')
                self.page_tables[name]=pages
            else:
                pages=self.page_tables[name]
                for p,s in zip(pos,slots):
                    if p//page==len(pages):pages.append(s//page)
                    if p%page!=s%page or pages[p//page]!=s//page:
                        raise RuntimeError('Unexpected native page remapping')
            cache=kv_cache.view(torch.float8_e4m3fn) if kv_cache.dtype==torch.uint8 else kv_cache
            if cache.dtype!=torch.float8_e4m3fn:
                raise RuntimeError('Expected native FP8 e4m3 cache')
            if name not in self.history_hashes:
                selected=torch.tensor(all_slots[:config['query_start']],device=cache.device)
                historical=cache.view(torch.uint8)[selected//page,:,selected%page,:]
                self.history_hashes[name]=sha(historical)
                if config.get('reference'):
                    self.reference_history[name]=self.history_hashes[name]
                elif self.reference_history[name]!=self.history_hashes[name]:
                    raise RuntimeError('Historical KV changed between paired arms before intervention')
                self.cache_geometry[name]=dict(shape=list(cache.shape), native_page_tokens=page,
                    strides=list(cache.stride()),native_layout=str(attn_metadata.kv_cache_layout),
                    mapped_tokens=length, history_tokens=config['query_start'],
                    candidate_physical_slots=[all_slots[p] for b in config['deleted_blocks']
                                              for p in range(b*32,(b+1)*32)],
                    k_scale=float(layer._k_scale.item()),v_scale=float(layer._v_scale.item()))
                del historical
            begin=next(i for i,p in enumerate(pos) if p>=config['query_start'])
            qp=pos[begin:]
            if config['mode']!='native':
                if self.workspace is None:
                    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
                    from vllm.v1.attention.backends.utils import get_flashinfer_layout_string
                    self.workspace=torch.empty(128*1024**2,dtype=torch.uint8,device=query.device)
                    self.wrapper=BatchPrefillWithPagedKVCacheWrapper(self.workspace,
                        get_flashinfer_layout_string(attn_metadata.kv_cache_layout),backend='fa2')
                if page!=32:
                    raise RuntimeError('Block exclusion requires the observed 32-token native kernel pages')
                removed=set(config['deleted_blocks'])
                if any((b+1)*32>config['query_start'] for b in removed):
                    raise RuntimeError('Attempt to remove current Query or future KV')
                retained=[physical for logical_page,physical in enumerate(pages) if logical_page not in removed]
                plan_key=(tuple(retained),tuple(pos),str(query.dtype))
                if self.plan_key!=plan_key:
                    def ints(values): return torch.tensor(values,device=query.device,dtype=torch.int32)
                    # Preserve the native Query batch shape. Exclude actual
                    # historical physical page IDs, keeping cached original RoPE
                    # keys and Query absolute positions. Only rows after the
                    # deletion boundary are used; earlier rows are discarded.
                    self.wrapper.plan(ints([0,count]),ints([0,len(retained)]),ints(retained),
                        ints([(length-1)%page+1]),24,4,256,page,
                        causal=True,q_data_type=torch.bfloat16,kv_data_type=cache.dtype,
                        o_data_type=torch.bfloat16,sm_scale=256**-.5)
                    self.plan_key=plan_key
                q=query[:count]
                if q.dtype==torch.float8_e4m3fn:
                    q=q.to(torch.bfloat16)*layer._q_scale
                else:
                    q=q.to(torch.bfloat16)
                native_view=cache.permute(*attn_metadata.kv_cache_layout.layer_view_order)
                custom=self.wrapper.run(q,native_view.split(256,-1),
                    k_scale=float(layer._k_scale.item()),v_scale=float(layer._v_scale.item()))
                if not skip_original:
                    native=output[begin:count].float()
                    self.native_relative.append(float((custom[begin:count].float()-native).norm()/(native.norm()+1e-8)))
                output[begin:count].copy_(custom[begin:count])
            self.record('attention_pre_gate_per_head',name,output[:count])
            if config.get('attention_scores') and name not in self.attention_mass:
                # Measured Query-to-KV associations from up to eight probe rows,
                # post-RoPE with the same scales and causal context as the model.
                indices=torch.tensor(all_slots,device=cache.device)
                k=cache.view(torch.uint8)[indices//page,:,indices%page,:256].view(torch.float8_e4m3fn).float()*layer._k_scale
                rows=list(range(begin,min(count,begin+8)))
                q=query[rows].float()
                if query.dtype==torch.float8_e4m3fn:q=q*layer._q_scale
                q=q.reshape(len(rows),4,6,256)
                logits=torch.einsum('tkhd,skd->tkhs',q,k)*256**-.5
                ks=torch.arange(length,device=query.device)
                valid=ks[None,:]<=torch.tensor([pos[i] for i in rows],device=query.device)[:,None]
                logits.masked_fill_(~valid[:,None,None,:],-torch.inf)
                mass=logits.softmax(-1).mean((0,1,2))
                padded=torch.nn.functional.pad(mass,(0,(-length)%32))
                self.attention_mass[name]=dict(query_positions=[pos[i] for i in rows],
                                               logical_block_mass=padded.reshape(-1,32).sum(-1).cpu().tolist())
            self.events+=1
            return result
        return forward

    def finish(self):
        torch.cuda.synchronize()
        result=dict(config=self.active, elapsed_seconds=time.perf_counter()-self.started,
            measurements=self.metrics, historical_kv_sha256=self.history_hashes,
            native_cache_geometry=self.cache_geometry, measured_attention_mass=self.attention_mass,
            relative_to_native_attention_max=max(self.native_relative,default=0.),
            attention_calls_after_boundary=self.events,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            original_positions_preserved=True, historical_kv_retained_for_shadow_audit=True,
            recurrent_state_policy='Original full-prefix recurrent history remains; it may retain candidate information.')
        self.active=None
        return result


class CleanupWorkerExtension:
    def cleanup_setup(self,config):
        if not hasattr(self,'_cleanup_measurement'):
            self._cleanup_measurement=Measurement(self.model_runner)
        return self._cleanup_measurement.configure(config)

    def cleanup_finish(self):
        return self._cleanup_measurement.finish()

    def cleanup_snapshot_prefix(self,length,path=None,reference_archive=None):
        """Read an owned completed request's still-unmodified physical cache.

        Called before any next request is admitted. No live service or foreign
        cache is involved; the isolated engine disables prefix reuse.
        """
        from pathlib import Path
        state=self._cleanup_measurement
        rows=[];hashes={}
        for name in state.attention_names:
            cache=state.modules[name].kv_cache
            if isinstance(cache,(list,tuple)):
                if len(cache)!=1:raise RuntimeError('Unexpected cache copies')
                cache=cache[0]
            page=cache.shape[2]
            slots=torch.tensor([state.slots[name][p] for p in range(length)],device=cache.device)
            value=cache.view(torch.uint8)[slots//page,:,slots%page,:].cpu()
            rows.append(value);hashes[name]=sha(value)
        tensor=torch.stack(rows)
        equal=None
        if reference_archive:
            ref=torch.load(reference_archive,map_location='cpu',weights_only=True)
            equal=bool(torch.equal(tensor,ref['kv'][:,:length]))
        if path:
            Path(path).parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(kv=tensor,names=state.attention_names,original_positions=list(range(length)),
                            scope='Native FP8 attention K/V; recurrent state is not in this file'),path)
        return dict(tokens=length,bytes=tensor.numel(),sha256=hashes,
                    original_prefix_recomputed_byte_equal=equal,path=path)
