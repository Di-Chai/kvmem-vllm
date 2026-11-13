"""Realistic proc/cgroup fixtures preserve memory-accounting scope and absence."""
import os
from pathlib import Path

import pytest

from vllm_kvmem.core.process_memory import sample_process_memory


def fixture(tmp_path):
    proc, root = tmp_path/'proc-self', tmp_path/'cgroups'
    proc.mkdir()
    group = root/'user.slice'/'worker.scope'
    group.mkdir(parents=True)
    (proc/'status').write_text('Name:\tworker\nVmRSS:\t2048 kB\nVmHWM:\t8192 kB\n'
        'RssAnon:\t1024 kB\nRssFile:\t512 kB\nRssShmem:\t512 kB\nVmLck:\t64 kB\n')
    (proc/'cgroup').write_text('0::/user.slice/worker.scope\n')
    (group/'memory.current').write_text('16777216\n')
    (group/'memory.max').write_text('max\n')
    (group/'memory.stat').write_text('anon 8388608\nfile 4194304\nshmem 524288\n'
        'file_dirty 4096\nfile_writeback 0\npgfault 999\n')
    return proc, root, group


def test_worker_rss_and_cgroup_file_charge_are_separate_nonadditive_observations(tmp_path):
    proc, root, _ = fixture(tmp_path)
    result = sample_process_memory(proc_self=proc, cgroup_root=root)
    assert not result['errors']
    assert result['pid'] == os.getpid()
    assert result['worker'] == dict(rss_bytes=2097152, lifetime_peak_rss_bytes=8388608,
        anon_rss_bytes=1048576, file_rss_bytes=524288, shmem_rss_bytes=524288, locked_bytes=65536)
    assert result['cgroup']['current_bytes'] == 16777216
    assert result['cgroup']['file_bytes'] == 4194304
    assert result['cgroup']['shmem_bytes'] == 524288
    assert result['cgroup']['limit_unbounded'] is True
    assert 'max_bytes' not in result['cgroup'] and 'pgfault_bytes' not in result['cgroup']
    assert 'not additive' in result['scope']


def test_missing_and_malformed_counters_never_become_fabricated_zeroes(tmp_path):
    proc, root, group = fixture(tmp_path)
    (proc/'status').write_text('VmRSS:\t12 MB\nVmHWM:\t-1 kB\nRssAnon:\t4 kB\n')
    (group/'memory.current').unlink()
    (group/'memory.max').write_text('33554432\n')
    (group/'memory.stat').write_text('anon invalid\nfile 2097152\n')
    result = sample_process_memory(proc_self=proc, cgroup_root=root)
    assert result['worker'] == dict(anon_rss_bytes=4096)
    assert 'current_bytes' not in result['cgroup']
    assert result['cgroup']['max_bytes'] == 33554432
    assert result['cgroup']['limit_unbounded'] is False
    assert 'anon_bytes' not in result['cgroup']
    assert result['cgroup']['file_bytes'] == 2097152
    assert len(result['errors']) == 4


@pytest.mark.parametrize('membership', ['0::/../../outside\n', '0::relative\n',
                                        '5:memory:/v1-only\n'])
def test_unsupported_membership_does_not_read_another_cgroup(tmp_path, membership):
    proc, root, _ = fixture(tmp_path)
    (proc/'cgroup').write_text(membership)
    result = sample_process_memory(proc_self=proc, cgroup_root=root)
    assert result['worker']['rss_bytes'] > 0
    assert result['cgroup'] == {}
    assert len(result['errors']) == 1


@pytest.mark.skipif(not Path('/proc/self/status').is_file(), reason='Linux procfs required')
def test_actual_worker_observation_without_gpu_or_allocator_import():
    result = sample_process_memory()
    assert result['pid'] == os.getpid()
    assert result['worker']['rss_bytes'] > 0
    assert result['worker']['lifetime_peak_rss_bytes'] >= result['worker']['rss_bytes']
    assert result['at'] > 0
