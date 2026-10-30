# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm import envs
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __contains__(self, request: object) -> bool:
        """Check membership without the ordered iteration's heap copy."""
        return request in self._heap

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class SJFRequestQueue(deque[Request], RequestQueue):
    """Shortest-job-first over waiting prefill admissions, with aging.

    Interactive small requests are admitted ahead of giant re-prefills so a
    new turn does not queue behind a neighbour's long-prefill burst (the HoL
    wait that dominates start latency under agent swarms). A request that
    has waited longer than the aging threshold is admitted FIFO ahead of
    the SJF order, so long prefills cannot starve.

    Selected by VLLM_SCHED_PREFILL_SJF=1 (the configured --scheduling-policy
    is ignored while the flag is set); the aging threshold is
    VLLM_SCHED_PREFILL_SJF_AGING_S seconds.

    Known v1 limitation: the sort key is the request's total prompt length,
    not the uncached remainder — a long session with a warm prefix cache is
    de-prioritized even though its re-prefill is small. Bounded by the queue
    depth; the true-uncached ordering needs admission-time radix lookups.
    """

    def __init__(self) -> None:
        super().__init__()
        self._arrival: dict[int, float] = {}
        # Requests explicitly re-queued at the front (prepend): admitted
        # before the SJF order, in prepend order.
        self._front: list[Request] = []

    def _arrival_of(self, request: Request) -> float:
        return self._arrival.get(id(request), time.time())

    def _is_aged(self, request: Request, now: float) -> bool:
        return now - self._arrival_of(request) >= envs.VLLM_SCHED_PREFILL_SJF_AGING_S

    def _selected(self) -> Request:
        """The request the policy admits next, without removing it."""
        if self._front:
            return self._front[0]
        if deque.__len__(self) <= 1:
            return self[0]
        now = time.time()
        aged = [(self._arrival_of(r), i, r)
                for i, r in enumerate(deque.__iter__(self))
                if self._is_aged(r, now)]
        if aged:
            # Bypass group: FIFO by arrival.
            return min(aged, key=lambda t: (t[0], t[1]))[2]
        return min(deque.__iter__(self),
                   key=lambda r: (r.num_tokens, self._arrival_of(r)))

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the SJF policy."""
        self._arrival[id(request)] = time.time()
        self.append(request)

    def pop_request(self) -> Request:
        """Pop the request the SJF policy admits next.

        Each request lives in exactly one structure (front list OR deque);
        the selected one is removed from its own.
        """
        request = self._selected()
        if self._front and self._front[0] is request:
            self._front.pop(0)
        else:
            self.remove(request)
        self._arrival.pop(id(request), None)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self._selected()

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue.

        Re-queued requests keep their original arrival time when known;
        unknown ones get a fresh aging clock. A request already in the deque
        is MOVED (it lives in exactly one structure).
        """
        self._arrival.setdefault(id(request), time.time())
        if request in self._front:
            self._front.remove(request)
        if request in self:
            deque.remove(self, request)
        self._front.insert(0, request)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        iteration order, so the iteration order of the source queue is
        preserved at the front of this queue.
        """
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        if request in self._front:
            self._front.remove(request)
        else:
            self.remove(request)
        self._arrival.pop(id(request), None)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue.

        Tolerates requests that are absent (the scheduler passes the same
        abort list to both the running and the waiting queues, and a request
        lives in at most one of them) — matching the FCFS queue contract.
        """
        doomed = {id(request) for request in requests}
        kept = [r for r in deque.__iter__(self) if id(r) not in doomed]
        self.clear()
        self.extend(kept)
        self._front[:] = [r for r in self._front if id(r) not in doomed]
        survivors = {id(r) for r in kept} | {id(r) for r in self._front}
        for rid in list(self._arrival):
            if rid not in survivors:
                del self._arrival[rid]

    def __bool__(self) -> bool:
        """Check if queue has any requests (front members included)."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return deque.__len__(self) + len(self._front)

    def __contains__(self, request: object) -> bool:
        """Membership across both structures."""
        return request in self._front or deque.__contains__(self, request)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue in the policy's admission order."""
        now = time.time()

        front_ids = {id(r) for r in self._front}

        def key(request: Request) -> tuple:
            arrival = self._arrival_of(request)
            if self._is_aged(request, now):
                return (0, arrival, 0)
            return (1, request.num_tokens, arrival)

        rest = [r for r in deque.__iter__(self) if id(r) not in front_ids]
        return iter([*self._front, *sorted(rest, key=key)])


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy.

    VLLM_SCHED_PREFILL_SJF=1 overrides the configured policy with the
    shortest-job-first waiting queue (aging via
    VLLM_SCHED_PREFILL_SJF_AGING_S).
    """
    if envs.VLLM_SCHED_PREFILL_SJF:
        return SJFRequestQueue()
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
