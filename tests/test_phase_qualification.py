import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('phase_qualification',
    Path(__file__).parents[1]/'scripts/qualify_agent_scheduling.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


@pytest.mark.parametrize('verify,arms', [(False, ['P1']), (True, ['A0']), (True, ['P1', 'A0'])])
def test_phase_correctness_cannot_silently_disable_oracles_or_coordinated_policy(tmp_path, verify, arms):
    args = NS(output=tmp_path, verify_restores=verify, cases=['agents'],
              phase_policy=True, arms=arms)
    with pytest.raises(ValueError, match='Phase qualification'):
        asyncio.run(qualification.main(args))


def test_phase_hot_long_wait_and_pressure_barrier_are_separate_before_gpu_launch(tmp_path):
    args = NS(output=tmp_path, verify_restores=True, cases=['agents'], phase_policy=True,
              arms=['P1', 'Ppressure'], retirement_barrier=True, distinct_users=True)
    with pytest.raises(ValueError, match='separate runs'):
        asyncio.run(qualification.main(args))
