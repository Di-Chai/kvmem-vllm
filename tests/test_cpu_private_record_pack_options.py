from copy import deepcopy
from dataclasses import replace
import pytest

from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from test_historical_options import settings as historical_options
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from vllm_kvmem.storage import make_shared_archive


def enabled():
    return dict(archive_backend='cpu',cpu_read_cache_bytes=1024**3,async_seal=True,
                cpu_direct_record_seal=True,cpu_seal_serial_allocate=True)


def test_default_remains_disabled():
    assert RuntimeOptions().cpu_seal_serial_allocate is False
    assert runtime().cpu_seal_serial_allocate is False


@pytest.mark.parametrize('value',[0,1,'true',None,[],{}])
def test_type_must_be_boolean(value):
    with pytest.raises(ValueError,match='must be boolean'):
        RuntimeOptions(cpu_seal_serial_allocate=value)
    with pytest.raises(ValueError,match='must be boolean'):
        runtime({**enabled(),'cpu_seal_serial_allocate':value})


def test_reject_incompatible_archive_modes():
    for extra in [dict(cpu_direct_record_seal=False),dict(cpu_seal_pack_workers=2)]:
        with pytest.raises(ValueError,match='cpu_seal_serial_allocate requires'):
            runtime({**enabled(),**extra})
    with pytest.raises(ValueError,match='without historical_gc'):
        replace(historical_options(),cpu_direct_record_seal=True,cpu_seal_serial_allocate=True)


def test_profile_worker_and_shared_archive_keep_the_explicit_flag(tmp_path):
    profile=model_profile(tmp_path)
    profile['kvmem'].update(enabled())
    original=deepcopy(profile)
    launch=build_launch(profile,tmp_path,tmp_path/'private-pack')
    options=runtime(worker_payload(launch))
    assert options.cpu_seal_serial_allocate and options.cpu_seal_pack_workers==1
    assert profile==original
    archive=make_shared_archive(options,tmp_path/'unused',4,8)
    assert archive.io_usage()['cpu_seal_serial_allocate']
    assert archive.io_usage()['seal_serial_allocate_batches_completed']==0
    assert archive.close()


def test_elastic_launch_keeps_the_flag(tmp_path):
    profile=elastic_profile(tmp_path)
    profile.update({key:value for key,value in enabled().items() if key!='cpu_read_cache_bytes'})
    cfg,_=elastic_launch(profile,tmp_path,tmp_path/'elastic')
    assert cfg['cpu_seal_serial_allocate']
    assert cfg['mean_k_backend_profile']['kvmem']['cpu_seal_serial_allocate']
