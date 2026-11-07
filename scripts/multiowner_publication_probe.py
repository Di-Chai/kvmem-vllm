"""Bounded multiowner Nsight observation; excluded from performance results."""
from prefill_pressure_probe import PressureProfiler, scheduled_rows


class MultiOwnerProfiler(PressureProfiler):
    def before_step(self, output):
        if not self.armed:return
        if self.started is not None and self.clock()-self.started>=self.config['observe_seconds']:
            self.finish('bounded_time_window')
            return
        rows=scheduled_rows(output)
        if not rows:return
        if any(not r['prefill'] for r in rows):
            self.finish('unexpected_decode')
            raise RuntimeError('Multiowner pressure diagnostic reached decode')
        if self.owner is None:self.owner=rows[0]['request_id']
        target=next((r for r in rows if r['request_id']==self.owner),None)
        if self.started is None:
            if target is None or target['computed']<self.config['begin_tokens']:return
            self.started=self.clock();self.profiler.start()
            if not self.is_running:
                self.finish('start_failed');raise RuntimeError('CUDA profiler did not start')
        if len(self.rows)>=64:
            self.finish('bounded_step_window')
            return
        self.rows.append(dict(at=self.clock(),owners=rows))


class _MultiOwnerMixin:
    def load_model(self,*args,**kwargs):
        import prefill_pressure_probe as module
        original=module.PressureProfiler
        module.PressureProfiler=MultiOwnerProfiler
        try:return super().load_model(*args,**kwargs)
        finally:module.PressureProfiler=original


def __getattr__(name):
    if name!='MultiOwnerPublicationWorker':raise AttributeError(name)
    from publication_bottleneck_probe import PublicationBottleneckWorker
    result=type(name,(_MultiOwnerMixin,PublicationBottleneckWorker),{'__module__':__name__})
    globals()[name]=result
    return result
