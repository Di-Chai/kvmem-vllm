"""Bounded source-copy/semantic-publication separation for accepted capture.

No score or checkpoint becomes visible when a DMA slot is released. The source
copy owns only immutable byte metadata; each owner retains its publication FIFO.
"""
import json
import time
from threading import current_thread


class EarlyCaptureFeedback:
    @property
    def early_capture_enabled(self):
        return getattr(self.options, 'early_capture_slot_release', False)

    def early_capture_packet(self, packet):
        # Decode retains its original tiny foreground / submit owner fence.
        return self.early_capture_enabled and packet.lane == 'bulk'

    def _feedback_bytes(self, packet, index):
        """Conservative independent tensor storage, frozen before score pop()."""
        return sum(self._feedback_storage_bytes(packet, index).values())

    def _feedback_storage_bytes(self, packet, index):
        rid = packet.request_ids[index]
        query = packet.query_inputs.get(rid)
        action = packet.action_inputs.get(rid)
        tensors = dict(query=query[1] if query is not None else None,
                       action=getattr(action, 'masses', None))
        return {name: t.untyped_storage().nbytes() if t is not None else 0
                for name, t in tensors.items()}

    def _early_limits(self):
        slots = len(self.downloads.buffers)
        return (getattr(self, '_early_owner_limit', 32), slots,
                self.options.early_capture_feedback_bytes)

    def _early_event(self, stage, packet, **values):
        # Bulk transitions are rare; decode keeps cumulative counters only.
        if packet.lane != 'bulk' or not hasattr(self, 'output'):
            return
        row = dict(at=time.monotonic(), wall_at=time.time(), stage=stage,
                   thread=current_thread().name, request_ids=packet.request_ids,
                   lane=packet.lane, slot=packet.download.slot,
                   generation=packet.source_generation, **values)
        with (self.output/'capture_feedback_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def _early_source_submitted(self, packet):
        """Under the publication lock, identify actual physical-slot reuse."""
        slots = self.__dict__.setdefault('_source_slot_generations', {})
        key = packet.lane, packet.download.slot
        packet.source_generation = slots[key] = slots.get(key, 0) + 1
        previous = [other for other in self.queue if other is not packet and
                    (other.lane, other.download.slot) == key and other.source_released and
                    len(other.published) < len(other.request_ids)]
        self._publication_metric('source_submitted_packets', 1, lane=packet.lane)
        if len(packet.request_ids) > 1:
            self._publication_metric('source_submitted_shared_packets', 1, lane=packet.lane)
        if previous:
            self._publication_metric('source_reuse_before_semantic_packets', 1, lane=packet.lane)
            if any(len(other.request_ids) > 1 for other in previous):
                self._publication_metric('shared_source_reuse_before_semantic_packets', 1, lane=packet.lane)
        self._early_event('source_submitted', packet,
                          pending_previous_generations=[other.source_generation for other in previous])

    def _try_release_source(self, packet):
        """Called under publication lock, once every owner has copied its bytes."""
        if packet.source_released:
            return True
        if len(packet.source_copied) != len(packet.request_ids):
            return False
        pending = set(range(len(packet.request_ids))) - packet.published
        credits = self.__dict__.setdefault('_feedback_credits', {})
        if pending:
            owners, packets, limit = self._early_limits()
            size = sum(packet.source_feedback_bytes.values())
            if (len(credits) >= packets or
                    sum(c[0] for c in credits.values()) + len(pending) > owners or
                    sum(c[1] for c in credits.values()) + size > limit):
                self._publication_metric('detach_budget_deferrals', 1, lane=packet.lane)
                return False
            credits[id(packet)] = (len(pending), size)
        if not self._download_pool(packet.lane).release(packet.download):
            credits.pop(id(packet), None)
            raise RuntimeError('Copied source still has an unfinished DMA fence')
        packet.source_released = True
        self._publication_metric('source_releases', 1, lane=packet.lane)
        if pending:
            self._publication_metric('source_releases_before_semantic', 1, lane=packet.lane)
        self._publication_metric('detached_feedback_bytes', sum(packet.source_feedback_bytes.values()),
                                 lane=packet.lane)
        self._early_event('source_released', packet,
                          feedback_bytes=sum(packet.source_feedback_bytes.values()),
                          semantic_pending=len(pending))
        self._publication_condition.notify_all()
        return True

    def _finish_early_packet(self, packet):
        if not packet.source_released:
            # Semantic completion never needs an extra detach credit.
            if not self._try_release_source(packet):
                raise RuntimeError('Completed semantic packet retained its source')
        self.__dict__.setdefault('_feedback_credits', {}).pop(id(packet), None)
        self._early_event('semantic_complete', packet)

    def _copy_slot_owner_prefix(self, lane):
        """Reclaim source slots without executing anyone's Query/Action scores.

        Copy can wait for its own DMA/archive capacity, as the original bounded
        source path did. A source whose score budget is exhausted stays pinned.
        """
        condition = self._publication_lock()
        with condition:
            self._raise_publication_error()
            blocked = set()
            choice = None
            for packet in self.queue:
                if packet.source_released:
                    blocked.update(rid for i, rid in enumerate(packet.request_ids)
                                   if i not in packet.published)
                    continue
                if packet.lane == lane and self._try_release_source(packet):
                    return True
                indices = []
                for i, rid in enumerate(packet.request_ids):
                    if i in packet.published:
                        continue
                    earlier = rid in blocked
                    blocked.add(rid)
                    if (packet.lane == lane and packet.confirmed and not earlier and
                            i not in packet.consuming and i not in packet.source_copied):
                        indices.append(i)
                if indices:
                    choice = packet, indices
                    packet.consuming.update(indices)
                    break
            if choice is None:
                return False
        packet, indices = choice
        try:
            copied = self._copy_claim_metadata(packet, indices, wait=True)
        except BaseException as error:
            with condition:
                self._publication_error = error
                condition.notify_all()
            raise
        with condition:
            packet.consuming.difference_update(indices)
            condition.notify_all()
        return bool(copied)

    def _ensure_early_download_slot(self, lane):
        pool = self._download_pool(lane)
        condition = self._publication_lock()
        started = time.perf_counter()
        try:
            while True:
                with condition:
                    self._raise_publication_error()
                    if len(pool.pending) < len(pool.buffers):
                        return
                self.start_cpu_scatter()
                if self._copy_slot_owner_prefix(lane):
                    continue
                with condition:
                    self._raise_publication_error()
                    if len(pool.pending) < len(pool.buffers):
                        return
                    owned = [p for p in self.queue if p.lane == lane and not p.source_released]
                    if owned and all(not p.confirmed for p in owned):
                        raise RuntimeError('Bounded capture slots contain only unconfirmed samples')
                    # Recheck credit releases/readiness after every wakeup. A
                    # timeout is only a poll, never fabricated publication.
                    condition.wait(.01)
        finally:
            self._publication_metric('copy_only_slot_wait_seconds', time.perf_counter()-started, lane=lane)
            self._publication_metric('copy_only_slot_wait_calls', 1, lane=lane)

    def _early_feedback_snapshot(self):
        credits = self.__dict__.setdefault('_feedback_credits', {})
        owners, packets, limit = self._early_limits()
        return dict(detached_packets=len(credits), detached_owners=sum(c[0] for c in credits.values()),
                    detached_bytes=sum(c[1] for c in credits.values()),
                    packet_limit=packets, owner_limit=owners, byte_limit=limit)

    def _sample_early_publication(self, progress):
        """Timestamped cumulative samples can bracket benchmark T0 and T1."""
        progress['publication_observed_at'] = time.monotonic()
        progress['publication_wall_at'] = time.time()
        progress['early_capture_feedback'] = self._early_feedback_snapshot()
        row = dict(at=progress['publication_observed_at'], wall_at=progress['publication_wall_at'],
                   publication_metrics=progress['publication_metrics'],
                   feedback=progress['early_capture_feedback'])
        with (self.output/'publication_samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
