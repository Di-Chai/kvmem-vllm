"""Opt-in diagnostic waves for reproducible full-input numerical checks.

This deliberately removes cross-owner readiness overlap. It is not installed
by serving or by throughput experiments. Original maintenance, KV admission,
physical planning, sampling and output limits remain authoritative.
"""
import json
import os
from ..core import settings as _settings


class CohortGate:
    def __init__(self, lengths):
        if (not lengths or any(type(n) is not int or n<=0 for n in lengths) or
                len(set(lengths))!=len(lengths)):
            raise ValueError('Diagnostic cohort requires unique positive prompt lengths')
        self.rank={n:i for i,n in enumerate(lengths)}
        self.owners={}
        self.allowed=set()

    def prepare(self,scheduler):
        from .request_plan import request_plan
        for request in scheduler.requests.values():
            n=request.num_prompt_tokens
            if n not in self.rank:continue
            previous=self.owners.setdefault(n,request)
            if previous is not request:
                raise RuntimeError('Diagnostic cohort repeats a prompt-length identity')
        self.allowed=set()
        if len(self.owners)!=len(self.rank):return
        historical=[];maintenance=[];live=[]
        for n in self.rank:
            request=self.owners[n]
            if request.is_finished():continue
            live.append(request)
            plan=request_plan(request,scheduler.query_policy.block_tokens,scheduler.query_policy.gen_budget)
            if plan is None or plan.query is None:
                raise RuntimeError('Diagnostic cohort requires an explicit final Query')
            if request.num_computed_tokens<plan.replay_begin:historical.append(request)
            if scheduler._phase_eligible(request) and scheduler._phase_refresh_only(request):
                maintenance.append(request)
        self.allowed.update(r.request_id for r in maintenance)
        if historical:
            if not maintenance and all(scheduler._phase_eligible(r) for r in historical):
                self.allowed.update(r.request_id for r in historical)
        elif live:
            # Query preparation may progress for all owners. Public Query and
            # generation execute in fixed input order, keeping decode shapes
            # out of the correctness comparison's timing-dependent batches.
            self.allowed.add(live[0].request_id)
        key=lambda r:self.rank.get(r.num_prompt_tokens,-1)
        scheduler.running.sort(key=key)
        for queue in scheduler._waiting_queues():
            ordered=sorted(queue,key=key);queue.clear();queue.extend(ordered)

    def permits(self,request):
        return request.num_prompt_tokens not in self.rank or request.request_id in self.allowed


def install():
    value=_settings.get('KVMEM_DIAGNOSTIC_PREFILL_COHORT_LENGTHS')
    if value is None:return
    lengths=json.loads(value)
    from .scheduler import KVMemScheduler
    if getattr(KVMemScheduler,'_diagnostic_cohort_gate_installed',False):return
    schedule,reserve=KVMemScheduler.schedule,KVMemScheduler._reserve_prefill_lookahead
    def controlled_schedule(scheduler,*args,**kwargs):
        gate=scheduler.__dict__.setdefault('_diagnostic_cohort_gate',CohortGate(lengths))
        gate.prepare(scheduler)
        return schedule(scheduler,*args,**kwargs)
    def controlled_reserve(scheduler,request,*args,**kwargs):
        gate=getattr(scheduler,'_diagnostic_cohort_gate',None)
        if gate is not None and not gate.permits(request):return 0
        return reserve(scheduler,request,*args,**kwargs)
    KVMemScheduler.schedule=controlled_schedule
    KVMemScheduler._reserve_prefill_lookahead=controlled_reserve
    KVMemScheduler._diagnostic_cohort_gate_installed=True
