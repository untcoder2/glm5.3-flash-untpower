# SJF waiting-queue tests: pytest-совместимы, но без фикстур — исполняются и
# run_without_pytest.py, и pytest, и напрямую. Контракт: VLLM_SCHED_PREFILL_SJF=1
# подменяет FCFS-очередь ожидания на shortest-job-first с aging-байпасом.
import os
from contextlib import contextmanager
from types import SimpleNamespace as NS

from vllm import envs
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    SJFRequestQueue,
    SchedulingPolicy,
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
        if clear is not None:
            clear()


def test_sjf_pops_shortest_first():
    q = SJFRequestQueue()
    big, small, mid = NS(num_tokens=400000), NS(num_tokens=2000), NS(num_tokens=5000)
    q.add_request(big)
    q.add_request(small)
    q.add_request(mid)
    assert q.pop_request() is small
    assert q.pop_request() is mid
    assert q.pop_request() is big


def test_aging_bypass_keeps_fifo_fairness():
    q = SJFRequestQueue()
    old = NS(num_tokens=400000)
    q.add_request(old)
    q._arrival[id(old)] -= 60  # переждал порог aging (30с по умолчанию)
    fresh1, fresh2 = NS(num_tokens=1), NS(num_tokens=2)
    q.add_request(fresh1)
    q.add_request(fresh2)
    assert q.pop_request() is old  # aged-байпас: FIFO по времени прихода
    assert q.pop_request() is fresh1
    assert q.pop_request() is fresh2


def test_prepend_request_goes_to_front():
    q = SJFRequestQueue()
    a, b = NS(num_tokens=100), NS(num_tokens=200)
    q.add_request(a)
    q.add_request(b)
    q.prepend_request(b)  # возврат вытесненного: ровно одна копия, в голове
    assert len(q) == 2
    assert q.peek_request() is b
    assert q.pop_request() is b
    assert q.pop_request() is a


def test_remove_requests_tolerates_absent_and_abort_mix():
    q = SJFRequestQueue()
    a, b, c = NS(num_tokens=1), NS(num_tokens=2), NS(num_tokens=3)
    q.add_request(a)
    q.add_request(b)
    # `c` в очереди не жил (уже running) — планировщик шлёт один список абортов
    # обеим очередям: отсутствие НЕ должно ронять движок (F1, PoC).
    q.remove_requests([a, c])
    assert len(q) == 1
    assert b in q
    assert a not in q


def test_front_member_counted_in_bool_and_len():
    # F3: член _front — единственный в очереди; __bool__ слепой к front вешал
    # вытесненные запросы навсегда.
    q = SJFRequestQueue()
    a = NS(num_tokens=1)
    q.add_request(a)
    assert q.pop_request() is a
    b = NS(num_tokens=5)
    q.prepend_request(b)
    assert len(q) == 1
    assert bool(q)
    assert q.pop_request() is b
    assert not bool(q)


def test_env_gate_factory_selects_sjf():
    with _env("VLLM_SCHED_PREFILL_SJF", "1"):
        assert isinstance(create_request_queue(SchedulingPolicy.FCFS), SJFRequestQueue)


def test_default_factory_stays_fcfs():
    with _env("VLLM_SCHED_PREFILL_SJF", "0"):
        assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
