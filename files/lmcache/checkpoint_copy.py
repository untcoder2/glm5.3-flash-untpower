# SPDX-License-Identifier: Apache-2.0
"""Engine-owned raw-page DMA for complete recurrent checkpoint bundles."""

# Standard
from collections.abc import Callable
from typing import Any
import json
import threading
import time

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.checkpoint_storage import checkpoint_page_groups
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    UnsafeCheckpointCopyError,
)
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)
from lmcache.v1.multiprocess.transfer_context.shm import ShmPoolMapping
from lmcache.v1.platform import torch_dev


class CheckpointPageCopier:
    """Copy only caller-pinned pages using existing worker CUDA and SHM contexts.

    Args:
        page_pool: Block-outermost, contiguous uint8 physical GPU page pool.
        layout: Expected address-free target/draft cache and auxiliary layout.
        initialize_layout: Validates and binds the runner's hidden-state views
            before a checkpoint can be restored, including before first forward.
        transfer: Existing pinned SHM byte mapping; caller-owned.
        capabilities: Checkpoint server format and SHM mapping identity.

    A completed call guarantees that all submitted DMA drained. GPU block pins,
    server leases and the registered SHM mapping must outlive that call.
    """

    def __init__(
        self,
        page_pool: torch.Tensor,
        layout: dict[str, Any],
        initialize_layout: Callable[[dict[str, Any]], None],
        transfer: ShmPoolMapping,
        capabilities: CheckpointCapabilities,
    ) -> None:
        if (
            page_pool.dtype != torch.uint8
            or page_pool.ndim != 2
            or not page_pool.is_contiguous()
            or page_pool.device.type != "cuda"
            or layout.get("page_bytes") != page_pool.shape[1]
        ):
            raise ValueError("Checkpoint copy requires a validated CUDA raw-page pool")
        initialize_layout(layout)
        transfer.checkpoint_slot_views(
            capabilities, CheckpointLeaseResponse("ready"), ()
        )
        self._pool = page_pool
        self._layout = json.loads(json.dumps(layout))
        self._transfer = transfer
        self._capabilities = capabilities
        self._stream = torch_dev.Stream(device=page_pool.device)
        self._lock = threading.Lock()
        # The DMA drain happens OUTSIDE the lock (event-based), so the next
        # lease can enqueue its own burst while this one drains on the stream.
        # Concurrency is bounded by the caller's executor (CheckpointTransfer
        # Worker workers=2 default), not here. The inflight counter guards the
        # remap path: it must never unmap pinned SHM bytes while a drain is
        # still running.
        self._inflight_cv = threading.Condition(self._lock)
        self._inflight = 0

    def __call__(
        self, job: CheckpointTransferJob, lease: CheckpointLeaseResponse
    ) -> None:
        """Validate the entire lease before DMA, then drain even a partial failure.

        Args:
            job: One rank's source/destination pages and optional producer event.
            lease: Server-pinned SHM byte slots for the entire manifest.

        Raises:
            ValueError: For incompatible layouts, page IDs or SHM descriptors;
                validation failures enqueue no copy.
            UnsafeCheckpointCopyError: If DMA cannot drain. Worker termination
                is required before reclaiming the transfer buffers.
        """
        payload = json.loads(job.manifest.payload)
        if payload.get("worker_layout") != self._layout:
            raise ValueError(
                "Checkpoint target/draft byte layout does not match this worker"
            )
        groups = checkpoint_page_groups(job.manifest)
        if len(groups) != len(job.block_ids):
            raise ValueError(
                "Checkpoint page groups differ from the destination bundle"
            )
        page_bytes = self._pool.shape[1]
        for group, ids in zip(groups, job.block_ids, strict=True):
            if group.page_bytes != page_bytes or len(group.positions) != len(ids):
                raise ValueError(
                    "Checkpoint page widths/counts differ from the manifest"
                )
            if any(
                type(block) is not int or not 0 < block < self._pool.shape[0]
                for block in ids
            ):
                raise ValueError(
                    "Checkpoint copy must not access null or out-of-range pages"
                )
        ids_flat = [block for ids in job.block_ids for block in ids]
        if len(ids_flat) != len(set(ids_flat)):
            raise ValueError("Checkpoint groups must own distinct physical pages")
        page_sizes = tuple(
            tuple(group.page_bytes for _ in group.positions) for group in groups
        )
        # One enqueue burst per lease prevents interleaving producer waits with
        # another task's copies on the shared stream. No model-thread barrier.
        # Views are borrowed under the same lock so a remap never runs while
        # another copy still holds bytes of the old mapping.
        done = None
        with (
            self._inflight_cv,
            torch_dev.device(self._pool.device),
            torch_dev.stream(self._stream),
        ):
            if not self._transfer.is_current():
                # The LMCache server restarted and recreated its pool. This
                # lease may belong to either server, so it is not trusted; later
                # leases come from the new server and use the new mapping.
                # Copies from earlier leases may still drain on the old
                # mapping; wait them out (bounded) before remapping. If a drain
                # is stuck past the deadline, remapping would unmap pinned SHM
                # bytes under live DMA, so refuse and require worker
                # termination instead.
                deadline = time.monotonic() + 30.0
                while self._inflight > 0 and time.monotonic() < deadline:
                    self._inflight_cv.wait(timeout=1.0)
                if self._inflight > 0:
                    raise UnsafeCheckpointCopyError(
                        "Checkpoint remap blocked by undrained copies; worker "
                        "termination required before reclaiming buffers"
                    )
                if self._transfer.is_current():
                    # Another worker already remapped while we waited; this
                    # lease still belongs to the old mapping.
                    raise ValueError(
                        "LMCache replaced its checkpoint SHM pool; lease not trusted"
                    )
                self._transfer.remap()
                raise ValueError(
                    "LMCache replaced its checkpoint SHM pool; lease not trusted"
                )
            buffers = self._transfer.checkpoint_slot_views(
                self._capabilities, lease, page_sizes
            )
            if job.direction == "RETRIEVE" and any(
                page is None for group in buffers for page in group
            ):
                raise ValueError("Checkpoint retrieval cannot omit payload pages")
            try:
                if job.producer_event is not None:
                    self._stream.wait_event(job.producer_event)
                for ids, pages in zip(job.block_ids, buffers, strict=True):
                    for block, page in zip(ids, pages, strict=True):
                        if page is None:
                            continue
                        if job.direction == "STORE":
                            page.copy_(self._pool[block], non_blocking=True)
                        else:
                            self._pool[block].copy_(page, non_blocking=True)
                ev = torch_dev.Event()
                ev.record(self._stream)
                done = ev  # assigned only after a successful record
                self._inflight += 1
            except BaseException:
                # Partial burst (or event/record failure): drain what was
                # enqueued before unwinding, so the lease cannot be released
                # under live DMA (legacy path).
                try:
                    self._stream.synchronize()
                except BaseException as exc:
                    raise UnsafeCheckpointCopyError(
                        "Checkpoint DMA could not drain"
                    ) from exc
                raise
        try:
            # Drain outside the lock: the next lease enqueues its own burst
            # while this burst's DMA runs on the stream. The call still
            # guarantees "returned = drained".
            done.synchronize()
        except BaseException as exc:
            raise UnsafeCheckpointCopyError(
                "Checkpoint DMA could not drain"
            ) from exc
        finally:
            with self._inflight_cv:
                self._inflight -= 1
                self._inflight_cv.notify_all()
