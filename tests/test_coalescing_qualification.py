import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec=importlib.util.spec_from_file_location('coalescing_qualification',Path(__file__).parents[1]/'scripts/qualify_agent_scheduling.py')
qualification=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualification)


@pytest.mark.parametrize('changes',[
    {'verify_restores':False}, {'phase_policy':True}, {'arms':['A0']},
])
def test_coalescing_cannot_bypass_byte_gate_or_use_phase_scheduler(tmp_path,changes):
    args=NS(output=tmp_path,verify_restores=True,cases=['agents'],prefill_coalescing=True,
        phase_policy=False,arms=['P1'],coalescing_mixed_start=False)
    for k,v in changes.items():setattr(args,k,v)
    with pytest.raises(ValueError,match='Coalescing qualification'):
        asyncio.run(qualification.main(args))


@pytest.mark.parametrize('changes',[
    {'prefill_coalescing':False}, {'arms':['Ppressure']}, {'retirement_barrier':True},
    {'context_tokens':65536}, {'output_tokens':64},
])
def test_mixed_start_cannot_accidentally_run_a_nonoverlapping_short_decode_case(tmp_path,changes):
    args=NS(output=tmp_path,verify_restores=True,cases=['agents'],prefill_coalescing=True,
        phase_policy=False,arms=['P1'],coalescing_mixed_start=True,
        retirement_barrier=False,context_tokens=32768,output_tokens=4096)
    for k,v in changes.items():setattr(args,k,v)
    with pytest.raises(ValueError,match='Coalescing mixed start'):
        asyncio.run(qualification.main(args))
