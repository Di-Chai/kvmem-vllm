from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[Pybind11Extension(
        "vllm_kvmem._native",
        ["csrc/bindings.cpp"] + [
            f"third_party/qw3/src/{name}.cpp" for name in
            ("kvmem_store", "kvmem_repository", "kvmem_repository_gc", "kvmem_gc")
        ],
        include_dirs=["third_party/qw3/include"],
        cxx_std=17,
        extra_compile_args=["-pthread"],
        extra_link_args=["-pthread"],
    )],
    cmdclass={"build_ext": build_ext},
)
