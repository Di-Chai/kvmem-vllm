"""Explicit, hash-checked loader fix for vLLM 0.30.0.

Adapted from vllm-project/vllm commit c723a831a81cb4ff89ea6d61b2d15109306ed1bc.
Only the explicit draft LoadConfig selection is backported. No kernel or model
math changes. Installed source files remain unchanged.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import sys

TARGET = "vllm.v1.worker.gpu.spec_decode.dflash.utils"
SOURCE_SHA256 = "53d2c3ad474d290d2c6df027446bbd80ac7447684e5242ab2c3c5337d2abd383"
OLD = "load_config=get_pp_safe_draft_load_config(vllm_config.load_config),"
NEW = """load_config=get_pp_safe_draft_load_config(
            speculative_config.draft_load_config
            if speculative_config.draft_load_config is not None
            else vllm_config.load_config
        ),"""
RUNNER_TARGET = "vllm.v1.worker.gpu.model_runner"
RUNNER_SHA256 = "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce"


def patched_source(source: bytes) -> str:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ImportError("Unrecognized vLLM DFlash loader; refusing unqualified compatibility patch")
    text = source.decode()
    if text.count(OLD) != 1:
        raise ImportError("Ambiguous draft loader patch")
    return text.replace(OLD, NEW)


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(patched_source(self.get_data(self.path)), self.path, "exec")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate pinned vLLM DFlash loader")
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable_draft_load_config_fix() -> None:
    if any(isinstance(f, _Finder) for f in sys.meta_path):
        return
    if importlib.metadata.version("vllm") != "0.30.0":
        raise RuntimeError("Draft loader fix only supports vLLM 0.30.0")
    if TARGET in sys.modules:
        raise RuntimeError("DFlash loader already imported; activate compatibility before loading vLLM")
    sys.meta_path.insert(0, _Finder())


class _RunnerLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path)
        if hashlib.sha256(source).hexdigest() != RUNNER_SHA256:
            raise ImportError("Unqualified V2 runner source; refusing runner extension")
        extension = "\nfrom vllm_kvmem.runner import make_probe_runner\nGPUModelRunner = make_probe_runner(GPUModelRunner)\n"
        return compile(source.decode() + extension, self.path, "exec")


class _RunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != RUNNER_TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate pinned V2 runner")
        spec.loader = _RunnerLoader(fullname, spec.origin)
        return spec


def enable_runner_probe() -> None:
    if any(isinstance(f, _RunnerFinder) for f in sys.meta_path):
        return
    if RUNNER_TARGET in sys.modules:
        raise RuntimeError("Runner extension must be enabled before V2 runner import")
    if importlib.metadata.version("vllm") != "0.30.0":
        raise RuntimeError("Runner extension only supports vLLM 0.30.0")
    sys.meta_path.insert(0, _RunnerFinder())


class _MetadataLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        from .metadata_patch import patch_metadata_source
        return compile(patch_metadata_source(self.get_data(self.path)), self.path, "exec")


class _MetadataFinder(importlib.abc.MetaPathFinder):
    target = "vllm.v1.worker.gpu.attn_utils"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError("Cannot locate pinned attention metadata builder")
        spec.loader = _MetadataLoader(fullname, spec.origin)
        return spec


def enable_window_metadata() -> None:
    if any(isinstance(f, _MetadataFinder) for f in sys.meta_path):
        return
    if _MetadataFinder.target in sys.modules:
        raise RuntimeError("Window metadata must be enabled before native module import")
    if importlib.metadata.version("vllm") != "0.30.0":
        raise RuntimeError("Window metadata only supports vLLM 0.30.0")
    sys.meta_path.insert(0, _MetadataFinder())
