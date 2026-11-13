"""Bounded experiments that amortize ordinary prefill's fixed model cost.

This offline policy never lowers the existing chunk size or gates decode.  Its
comparison uses total model milliseconds per public prefill token, including
the other requests' decode work, sampling and MTP; maintenance is excluded.  A
better ratio therefore does not promise better decode latency or throughput.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
import statistics
import time


@dataclass
class _Bank:
    epoch: int
    started: float
    samples: deque = field(default_factory=lambda: deque(maxlen=32))
    issued: int = 0


class OfflinePrefillCoalescer:
    """Try at most three larger chunks per retained 60-second bank epoch.

    Evicting an exact batch/context bank requires baseline relearning.  This
    bounded calibration is not a global rolling-minute quota or an SLO bound.

    ``cap`` is called once for the single ordinary prefill candidate.  When it
    grants a calibration chunk, immediately save ``last_authorization_id``.
    After native scheduling call ``reconcile(id, actual_tokens=...)`` (zero for
    an unsubmitted candidate).  Copy that id to the measured batch's
    ``coalescing_authorization_id`` so feedback can verify the *actual* shape.
    Missing reconciliation is conservative: the three grants remain consumed.
    """
    feedback_max_age = 5.0
    retention = 60.0
    max_keys = 64
    calibration_limit = 3
    required_samples = 3
    minimum_gain = 0.05
    maximum_model_growth = 2.0
    maximum_sample_spread = 1.5

    def __init__(self, baseline_tokens, maximum_tokens, block_tokens, *, clock=time.monotonic):
        if any(type(v) is not int or v <= 0
               for v in (baseline_tokens, maximum_tokens, block_tokens)):
            raise ValueError('Prefill sizes must be positive integers')
        if (baseline_tokens % block_tokens or maximum_tokens % block_tokens or
                not baseline_tokens <= maximum_tokens <= 2 * baseline_tokens):
            raise ValueError('Prefill sizes must be aligned and maximum at most twice baseline')
        self.baseline, self.maximum, self.block = baseline_tokens, maximum_tokens, block_tokens
        self.clock = clock
        self.banks = OrderedDict()
        self.sequence = 0
        self._epoch = self._authorization = 0
        self._receipts = {}
        self.last_authorization_id = None
        self.observations = self.ignored = self.calibration_grants = 0
        self.approved_grants = self.released_grants = self.shape_mismatches = 0
        self.discarded_grants = 0
        self.last_decision = 'baseline: no matching observations'

    @staticmethod
    def _key(decode_requests, prefill_requests, decode_context_bucket, prefill_context_bucket):
        if (type(decode_requests) is not int or decode_requests < 0 or
                type(prefill_requests) is not int or prefill_requests < 0):
            raise ValueError('Request counts must be nonnegative integers')
        for value in (decode_context_bucket, prefill_context_bucket):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError('Context buckets must be positive integers or None')
        if (prefill_requests != 1 or prefill_context_bucket is None or
                (decode_requests and decode_context_bucket is None)):
            return None
        return (decode_requests, prefill_requests,
                decode_context_bucket if decode_requests else None, prefill_context_bucket)

    def _drop(self, key):
        self.banks.pop(key, None)
        for receipt, (owner, _, _) in list(self._receipts.items()):
            if owner == key:
                del self._receipts[receipt]

    def _prune(self, now):
        for key, bank in list(self.banks.items()):
            if now - bank.started >= self.retention:
                self._drop(key)
            else:
                bank.samples = deque((sample for sample in bank.samples
                                      if now - sample[0] < self.retention), maxlen=32)

    def _bank(self, key, now, *, create=False):
        bank = self.banks.get(key)
        if bank is None and create:
            self._epoch += 1
            bank = self.banks[key] = _Bank(self._epoch, now)
            if len(self.banks) > self.max_keys:
                self._drop(next(iter(self.banks)))
        if bank is not None:
            self.banks.move_to_end(key)
        return bank

    @staticmethod
    def _finite(value):
        return (isinstance(value, (float, int)) and not isinstance(value, bool) and
                math.isfinite(value))

    def _group(self, tokens):
        if .9 * self.baseline <= tokens <= self.baseline:
            return 'baseline'
        if self.maximum > self.baseline and .9 * self.maximum <= tokens <= self.maximum:
            return 'burst'
        return None

    def _assessment(self, bank):
        baseline = [(tokens, ms) for _, group, tokens, ms in bank.samples if group == 'baseline']
        burst = [(tokens, ms) for _, group, tokens, ms in bank.samples if group == 'burst']
        report = dict(baseline_samples=len(baseline), burst_samples=len(burst),
                      calibration_grants=bank.issued)
        if len(baseline) < self.required_samples:
            return 'need_baseline', report
        base_rates = [ms / tokens for tokens, ms in baseline]
        if max(base_rates) > self.maximum_sample_spread * min(base_rates):
            return 'unstable_baseline', report
        base_ms = statistics.median(ms for _, ms in baseline)
        base_rate = statistics.median(base_rates)
        report.update(baseline_model_ms=base_ms, baseline_model_ms_per_token=base_rate)
        if len(burst) < self.required_samples:
            return ('calibrating' if bank.issued < self.calibration_limit else 'awaiting_feedback'), report
        burst_rates = [ms / tokens for tokens, ms in burst]
        burst_ms = statistics.median(ms for _, ms in burst)
        burst_rate = statistics.median(burst_rates)
        report.update(burst_model_ms=burst_ms, burst_model_ms_per_token=burst_rate,
                      measured_gain=1.0 - burst_rate / base_rate)
        if max(burst_rates) > self.maximum_sample_spread * min(burst_rates):
            return 'unstable_burst', report
        if burst_rate > (1.0 - self.minimum_gain) * base_rate:
            return 'insufficient_gain', report
        if burst_ms > self.maximum_model_growth * base_ms:
            return 'model_step_growth', report
        return 'approved', report

    def reconcile(self, authorization_id, *, actual_tokens, **actual_shape):
        """Settle an unused/trimmed/mismatched grant once; safe to repeat.

        Optional actual_shape contains all four ``cap`` shape arguments.  A
        full submission keeps its receipt until measured feedback arrives.
        Only execution of at most the baseline refunds a calibration grant.
        Any extra executed work consumes it, even if too short to train or
        executed in a different actual shape.
        """
        if type(authorization_id) is not int or authorization_id not in self._receipts:
            return False
        if type(actual_tokens) is not int or actual_tokens < 0:
            raise ValueError('Actual prefill tokens must be a nonnegative integer')
        key, epoch, _ = self._receipts[authorization_id]
        mismatch = bool(actual_shape) and self._key(**actual_shape) != key
        if self._group(actual_tokens) != 'burst' or mismatch:
            del self._receipts[authorization_id]
            bank = self.banks.get(key)
            refund = actual_tokens <= self.baseline
            if refund and bank is not None and bank.epoch == epoch:
                bank.issued -= 1
            self.released_grants += int(refund)
            self.discarded_grants += int(not refund)
            self.shape_mismatches += int(mismatch)
            return True
        self._receipts[authorization_id] = (key, epoch, True)
        return False

    def observe(self, rows):
        now = self.clock()
        self._prune(now)
        for row in rows:
            sequence = row.get('sequence')
            if type(sequence) is not int or sequence <= self.sequence:
                self.ignored += 1
                continue
            self.sequence = sequence
            submitted, ms, tokens = (row.get('submitted_at'), row.get('model_ms'),
                                     row.get('prefill_tokens'))
            if (row.get('ordinary_prefill_only') is not True or
                    not self._finite(submitted) or not 0 <= now - submitted <= self.feedback_max_age or
                    not self._finite(ms) or ms <= 0 or type(tokens) is not int or tokens <= 0):
                self.ignored += 1
                continue
            try:
                key = self._key(*(row.get(name) for name in ('decode_requests', 'prefill_requests',
                    'decode_context_bucket', 'prefill_context_bucket')))
            except ValueError:
                key = None
            if key is None or self.maximum == self.baseline:
                self.ignored += 1
                continue
            group = self._group(tokens)
            receipt = row.get('coalescing_authorization_id')
            authorization = self._receipts.get(receipt) if type(receipt) is int else None
            if authorization is not None and authorization[0] != key:
                self.reconcile(receipt, actual_tokens=tokens,
                    **dict(zip(('decode_requests', 'prefill_requests',
                                'decode_context_bucket', 'prefill_context_bucket'), key)))
                self.ignored += 1
                continue
            if group is None:
                if authorization is not None:
                    self.reconcile(receipt, actual_tokens=tokens)
                self.ignored += 1
                continue
            bank = self._bank(key, now, create=group == 'baseline')
            if bank is None:
                self.ignored += 1
                continue
            if group == 'burst':
                approved = self._assessment(bank)[0] == 'approved'
                if authorization is None and not approved:
                    self.ignored += 1
                    continue
                if authorization is not None:
                    if authorization[1] != bank.epoch:
                        self.ignored += 1
                        continue
                    del self._receipts[receipt]
            elif authorization is not None:
                self.reconcile(receipt, actual_tokens=tokens)
            bank.samples.append((submitted, group, tokens, ms))
            self.observations += 1

    def cap(self, *, decode_requests, prefill_requests, decode_context_bucket,
            prefill_context_bucket, remaining_tokens):
        self.last_authorization_id = None
        if type(remaining_tokens) is not int or remaining_tokens < 0:
            raise ValueError('Remaining prefill tokens must be a nonnegative integer')
        key = self._key(decode_requests, prefill_requests,
                        decode_context_bucket, prefill_context_bucket)
        now = self.clock()
        self._prune(now)
        if key is None or self.maximum == self.baseline or remaining_tokens < 2 * self.baseline:
            self.last_decision = 'baseline: ineligible shape or short remaining input'
            return self.baseline
        bank = self._bank(key, now)
        if bank is None:
            self.last_decision = 'baseline: no matching observations'
            return self.baseline
        state, _ = self._assessment(bank)
        if state == 'approved':
            self.approved_grants += 1
            self.last_decision = 'coalesced: measured model cost improved'
            return self.maximum
        if state == 'calibrating':
            self._authorization += 1
            self.last_authorization_id = self._authorization
            self._receipts[self._authorization] = (key, bank.epoch, False)
            bank.issued += 1
            self.calibration_grants += 1
            self.last_decision = 'coalesced: bounded calibration'
            return self.maximum
        self.last_decision = 'baseline: ' + state
        return self.baseline

    def snapshot(self):
        self._prune(self.clock())
        return dict(baseline_tokens=self.baseline, maximum_tokens=self.maximum,
            cost_scope='total model_ms/public prefill token; includes decode, sampling and MTP; excludes maintenance',
            decode_latency_tradeoff=True, feedback_max_age_seconds=self.feedback_max_age,
            retention_seconds=self.retention, observations=self.observations, ignored=self.ignored,
            calibration_grants=self.calibration_grants, approved_grants=self.approved_grants,
            released_grants=self.released_grants, shape_mismatches=self.shape_mismatches,
            discarded_grants=self.discarded_grants,
            pending_authorizations=len(self._receipts), last_decision=self.last_decision,
            banks=[dict(key=list(key), epoch=bank.epoch, state=self._assessment(bank)[0],
                        **self._assessment(bank)[1]) for key, bank in self.banks.items()])
