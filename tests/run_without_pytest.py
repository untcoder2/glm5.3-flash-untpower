# Fallback runner: executes the parametrized tests without pytest installed.
# Usage (inside the overlay image): python3 /tests/run_without_pytest.py
import sys
import types

try:
    import pytest  # noqa: F401
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    stub = types.ModuleType("pytest")

    class _Mark:
        def parametrize(self, *args, **kwargs):
            def deco(fn):
                fn._parametrize = args
                return fn
            return deco

    class MonkeyPatch:  # type annotation only in the test signatures
        pass

    stub.mark = _Mark()
    stub.MonkeyPatch = MonkeyPatch
    sys.modules["pytest"] = stub

import torch  # noqa: E402
import test_split_ckv_glm5next as t  # noqa: E402


class _MP:
    def __init__(self):
        self._records = []

    def setattr(self, obj, name, value):
        had = name in vars(obj)
        old = getattr(obj, name, None)
        setattr(obj, name, value)
        self._records.append((obj, name, had, old))

    def undo(self):
        for obj, name, had, old in reversed(self._records):
            if had:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
        self._records.clear()


failures = []

# 1. Heuristic test: expand the parametrize cases manually.
cases = [
    (True, 8, 1, 8192, True),
    (True, 0, 1, 8192, False),
    (True, 8, 0, 0, False),
    (True, 8, 1, 8, False),
    (True, 8, 1, 600000, False),
    (False, 8, 1, 8192, False),
]
for case in cases:
    try:
        t.test_b12x_split_ckv_gather_needs_a_mixed_sparse_glm_batch(*case)
        print(f"OK  heuristic {case}")
    except AssertionError as e:
        failures.append(("heuristic", case, str(e)))
        print(f"FAIL heuristic {case}: {e}")

# 2. Builder tests: both families, all dcp_rank x split combinations.
for family, fn in (
    ("glm_dsa", t.test_glm_dsa_builder_gathers_only_the_prefill_requests_of_a_mixed_batch),
    ("glm5next", t.test_glm5next_builder_gathers_only_the_prefill_requests_of_a_mixed_batch),
):
    for dcp_rank in (0, 1):
        for split in (False, True):
            mp = _MP()
            try:
                fn(mp, dcp_rank, split)
                print(f"OK  builder {family} rank={dcp_rank} split={split}")
            except Exception as e:  # noqa: BLE001
                failures.append((family, dcp_rank, split, repr(e)))
                print(f"FAIL builder {family} rank={dcp_rank} split={split}: {e!r}")
            finally:
                mp.undo()

print(f"\nTOTAL: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)")
sys.exit(1 if failures else 0)
