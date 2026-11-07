# SJF v2 admission-key tests: the queue sorts by the uncached remainder (a
# read-only radix probe injected by the scheduler) instead of the total prompt
# length. pytest-совместимы, но без фикстур — исполняются и run_without_pytest.py,
# и pytest, и напрямую. Контракт: исключение зонда обязано падать в fallback по
# полной длине, никогда не роняя admission.
import os
import time
from contextlib import contextmanager

from vllm import envs
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    SJFRequestQueue,
    create_request_queue,
)


@contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    clear = getattr(getattr(envs, "__getattr__", None), "cache_clear", None)
    if clear is not None:
        clear()  # enable_envs_cache() мог уже заморозить значения
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class FakeReq:
    def __init__(self, n, rid=None):
        self.num_tokens = n
        self.rid = rid or f"r{n}"

    def __repr__(self):
        return f"FakeReq({self.num_tokens})"


warm, small = FakeReq(252_692), FakeReq(5_000)
probe = {id(warm): 58, id(small): 5_000}  # warm-fat: 58 tokens of real work


def probe_fn(request):
    return probe[id(request)]


def test_v2_warm_fat_first():
    # Regression for the live incident: a warm fat session (58 tokens of work)
    # must not queue behind a cold small one.
    q = SJFRequestQueue(cache_probe=probe_fn)
    q.add_request(warm)
    q.add_request(small)
    assert q.pop_request() is warm
    assert q.pop_request() is small


def test_fallback_no_probe():
    # Without a probe the v1 behaviour is preserved (total prompt length).
    q = SJFRequestQueue(cache_probe=None)
    q.add_request(warm)
    q.add_request(small)
    assert q.pop_request() is small


def test_fallback_probe_exception():
    def broken(request):
        raise RuntimeError("radix unavailable")

    q = SJFRequestQueue(cache_probe=broken)
    q.add_request(warm)
    q.add_request(small)
    assert q.pop_request() is small


def test_aged_bypass():
    old = FakeReq(100)
    q = SJFRequestQueue(cache_probe=probe_fn)
    q.add_request(small)
    q.add_request(warm)
    q.add_request(old)
    q._arrival[id(old)] = time.time() - 10  # add_request resets the clock
    assert q.pop_request() is old


def test_prepend_front():
    front = FakeReq(999_999)
    q = SJFRequestQueue(cache_probe=probe_fn)
    q.add_request(small)
    q.prepend_request(front)
    assert q.pop_request() is front


def test_remove_tolerates_absent():
    q = SJFRequestQueue(cache_probe=probe_fn)
    q.add_request(small)
    q.add_request(warm)
    q.remove_requests([FakeReq(1)])  # abort list may name a foreign request
    assert len(q) == 2


def test_iter_matches_pop_order():
    q = SJFRequestQueue(cache_probe=probe_fn)
    q.add_request(small)
    q.add_request(warm)
    assert list(q)[0] is warm


def test_factory_with_probe():
    with _env("VLLM_SCHED_PREFILL_SJF", "1"):
        q = create_request_queue(SchedulingPolicy.FCFS, cache_probe=probe_fn)
    assert isinstance(q, SJFRequestQueue)
