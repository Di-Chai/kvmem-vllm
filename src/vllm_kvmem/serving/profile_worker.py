"""Diagnostic native worker: annotate launches without installing KVMem hooks."""
from vllm.v1.worker.gpu_worker import Worker


class NativeProfileWorker(Worker):
    def initialize_from_config(self, *args, **kwargs):
        super().initialize_from_config(*args, **kwargs)
        from .profile_scopes import install_profile_scopes
        install_profile_scopes(self.model_runner)
