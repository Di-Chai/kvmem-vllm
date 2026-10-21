"""Startup-only compiler audit for a frozen diagnostic serving process.

The native worker keeps its normal model loading, compilation and execution.
Only initialization observes actual Inductor launches; the audit must restore
its original method before this worker can become ready for public requests.
Importing the lifecycle mixin does not import Torch or vLLM.
"""
import os
from pathlib import Path


RUN_DIRECTORY_ENV = 'PREFILL_INDUCTOR_RUN_DIR'


def startup_audit_path():
    value = os.environ.get(RUN_DIRECTORY_ENV)
    if not value or '\x00' in value:
        raise ValueError(f'{RUN_DIRECTORY_ENV} must name an existing absolute run directory')
    directory = Path(value)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError(f'{RUN_DIRECTORY_ENV} must name an existing absolute run directory')
    return directory.resolve() / f'inductor-startup-{os.getpid()}.json'


class StartupAuditWorkerMixin:
    """Observe initialization only; no forward, sampling or replay wrappers."""

    def _close_inductor_startup_audit(self, success):
        handle = getattr(self, '_inductor_startup_audit_handle', None)
        if handle is None:
            raise RuntimeError('Inductor startup audit is not active')
        # close() restores the patched method even if writing/checking its
        # receipt fails. Never invoke it twice or retain audit objects in the
        # worker after the initialization boundary.
        self._inductor_startup_audit_handle = None
        handle.close(success=success)

    def _fail_inductor_startup_audit(self, error):
        try:
            self._close_inductor_startup_audit(False)
        except BaseException as cleanup_error:
            # Keep the original model/compilation failure, while retaining the
            # audit failure as evidence. A failed startup never serves work.
            error.add_note(f'Inductor startup audit cleanup failed: {cleanup_error!r}')

    def load_model(self, *args, **kwargs):
        if getattr(self, '_inductor_startup_audit_started', False):
            raise RuntimeError('Inductor startup audit cannot be restarted')
        from prefill_inductor_control import install_launch_audit

        output = startup_audit_path()
        handle = install_launch_audit(output)
        self._inductor_startup_audit_started = True
        self._inductor_startup_audit_handle = handle
        try:
            return super().load_model(*args, **kwargs)
        except BaseException as error:
            self._fail_inductor_startup_audit(error)
            raise

    def compile_or_warm_up_model(self, *args, **kwargs):
        if getattr(self, '_inductor_startup_audit_handle', None) is None:
            raise RuntimeError('Native warmup requires an active Inductor startup audit')
        try:
            result = super().compile_or_warm_up_model(*args, **kwargs)
        except BaseException as error:
            self._fail_inductor_startup_audit(error)
            raise
        self._close_inductor_startup_audit(True)
        return result


def __getattr__(name):
    if name == 'SeededNativeProbeWorker':
        from native_prefill_shape_probe import NativeProbeWorker

        class SeededNativeProbeWorker(StartupAuditWorkerMixin, NativeProbeWorker):
            pass

        globals()[name] = SeededNativeProbeWorker
        return SeededNativeProbeWorker
    if name != 'SeededAuditWorker':
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker

    class SeededAuditWorker(StartupAuditWorkerMixin, Worker):
        pass

    # vLLM resolves --worker-cls by module and name in the worker process.
    globals()[name] = SeededAuditWorker
    return SeededAuditWorker
