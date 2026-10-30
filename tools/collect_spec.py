#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Коллектор telemetry спекулятивного декодирования (spec decode) для дашборда.

Стадия 1, пассивная: только читает /metrics инференса, ничего в движке не
меняет, рестартов нет. Источник: http://127.0.0.1:1234/metrics (vLLM, без
auth; с Windows для смоука — DASH_SPEC_BASE=http://<engine-host>:1234).
Серии (лейблы engine="0", model_name="qwen3.8-flash-next"; суммируем по всем
лейблам — движок один, тот же приём, что в collect_server):
    vllm:spec_decode_num_drafts_total                    — draft-шагов
    vllm:spec_decode_num_draft_tokens_total              — проверенных draft-токенов
    vllm:spec_decode_num_accepted_tokens_total           — принятых токенов
    vllm:spec_decode_num_accepted_tokens_per_pos_total   — принятых по позиции
    vllm:spec_decode_num_draft_tokens_per_pos_total      — черновиков по позиции
Гейджи *_created по имени не совпадают и игнорируются сами собой.

Секция "spec" (CONTRACT.md). Rates — по накопленным счётчикам (baseline +
текущий бут, тот же fold через рестарты, что у collect_server):
    accept_rate = accepted / draft_tokens * 100   (% принятых draft-токенов)
    accept_len  = accepted / drafts               (ср. принятых токенов на шаг)
    pos_rates   = accepted_pos[p] / draft_pos[p] * 100 по каждой позиции

Персистентность (тот же паттерн, что server_state/server_gen):
    data/server_spec_state.json — {boot_id, last, baseline, updated_at};
      рестарт движка (boot_id сменился или счётчик уменьшился — vLLM сбрасывает
      counters при рестарте) => last сворачивается в baseline; дельты наружу
      никогда не отрицательные; первый запуск = только baseline.
    data/server_spec_buckets.json — почасовые дельты, ключи UTC:
      {"YYYY-MM-DD": {"<HH>:00-utc": {drafts, accepted, draft_pos, accepted_pos}}};
      только аддитивный рост, ничего не чистим (стадия 1).
Критическая секция state+buckets — под flock (data/server_spec.lock):
writers — tick_counter.py (каждые 30 мин) и cron refresh (10:00/20:00).
Частота сэмплов: бакеты питает тик tick_counter.py, отдельной cron-строки нет.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = os.environ.get("DASH_SPEC_BASE", "http://127.0.0.1:1234")
TIMEOUT_S = 8.0
HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "data" / "server_spec_state.json"
BUCKETS_PATH = HERE / "data" / "server_spec_buckets.json"
LOCK_PATH = HERE / "data" / "server_spec.lock"

# Контейнер движка для boot_id — тот же, что DASH_GEN_CONTAINER в collect_server
# (glm-shadow с 28.10). Отдельная переменная намеренно: коллекторы не связаны,
# падение/правка одного не тянет второй; при переезде движка обновить обе.
ENGINE_CONTAINER = os.environ.get("DASH_SPEC_CONTAINER", "glm-shadow")

# Плоские счётчики: метрика -> ключ; сумма по всем лейблам (движок один).
SPEC_COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
}
# Позиционные серии: метрика -> ключ списка; позиция — лейбл position="N".
SPEC_POS = {
    "vllm:spec_decode_num_accepted_tokens_per_pos_total": "accepted_pos",
    "vllm:spec_decode_num_draft_tokens_per_pos_total": "draft_pos",
}
MAX_POS = 8  # страховка от мусорного position="999"

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^{}]*)\})?"
    r"\s+(?P<value>[-+0-9.eE]+|NaN|Inf|-Inf)"
    r"(?:\s+\d+)?\s*$"
)
_POS_LABEL = re.compile(r'\bposition="(\d+)"')


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _num(raw: str) -> float | None:
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None  # NaN/Inf -> None


def _parse_metrics(text: str) -> tuple[dict, dict]:
    """(counters, pos) из Prometheus-текста; счётчики int, позиции — списки."""
    flat: dict[str, float] = {}
    pos_acc: dict[tuple[str, int], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, value = m["name"], _num(m["value"])
        if value is None:
            continue
        if name in SPEC_COUNTERS:
            flat[name] = flat.get(name, 0.0) + value
        elif name in SPEC_POS:
            pm = _POS_LABEL.search(m["labels"] or "")
            if pm:
                p = int(pm.group(1))
                if p < MAX_POS:
                    pos_acc[(name, p)] = pos_acc.get((name, p), 0.0) + value

    counters = {key: int(round(flat[n])) for n, key in SPEC_COUNTERS.items()
                if n in flat}
    pos: dict[str, list[int]] = {}
    for n, key in SPEC_POS.items():
        vals = [(p, v) for (nm, p), v in pos_acc.items() if nm == n]
        if vals:
            arr = [0] * (max(p for p, _ in vals) + 1)
            for p, v in vals:
                arr[p] = int(round(v))
            pos[key] = arr
    return counters, pos


def _snapshot() -> tuple[dict | None, dict]:
    """(counters, pos) с /metrics; (None, {}) — движок недоступен/ошибка сети."""
    try:
        req = urllib.request.Request(BASE + "/metrics",
                                     headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError, ValueError, TimeoutError):
        return None, {}
    return _parse_metrics(text)


# --- накопительные счётчики через рестарты (паттерн collect_server._fold) ------

def _boot_id() -> str | None:
    """Идентичность контейнера движка (только на хосте, где есть docker)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", ENGINE_CONTAINER or ""):
        return None
    try:
        import subprocess
        r = subprocess.run(
            ["docker", "inspect", ENGINE_CONTAINER,
             "--format", "{{.Id}} {{.State.StartedAt}}"],
            capture_output=True, text=True, timeout=8)
        s = (r.stdout or "").strip()
        return s or None
    except Exception:
        return None


def _load_json_dict(path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _atomic_write_json(path, obj: dict) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _i(v) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _norm_cur(obj) -> dict:
    """Снапшот/запись state -> нормализованный dict (скаляры + позиции)."""
    if not isinstance(obj, dict):
        return {"drafts": 0, "draft_tokens": 0, "accepted": 0,
                "draft_pos": [], "accepted_pos": []}
    arr = lambda v: [_i(x) for x in v] if isinstance(v, list) else []
    return {"drafts": _i(obj.get("drafts")),
            "draft_tokens": _i(obj.get("draft_tokens")),
            "accepted": _i(obj.get("accepted")),
            "draft_pos": arr(obj.get("draft_pos")),
            "accepted_pos": arr(obj.get("accepted_pos"))}


def _vec_add(a: list, b: list) -> list:
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
            for i in range(n)]


def _vec_sub(cur: list, prev: list) -> list:
    """cur - prev поэлементно, без отрицательных (пухлость/сбой — зажим в 0)."""
    return [max(0, (cur[i] if i < len(cur) else 0) - (prev[i] if i < len(prev) else 0))
            for i in range(max(len(cur), len(prev)))]


def _any(cur: dict) -> bool:
    return bool(cur["drafts"] or cur["draft_tokens"] or cur["accepted"]
                or any(cur["draft_pos"]) or any(cur["accepted_pos"]))


def _update(cur: dict) -> tuple[dict, dict]:
    """Дельта с прошлого тика -> state + buckets под flock. (итог, baseline).

    Движок молчал/серий нет (cur нулевой) — состояние не трогаем, чтобы не
    создавать фальшивый «рестарт» при появлении серий. Первый запуск —
    baseline only, без бакетной записи. Рестарт: last уходит в baseline,
    дельта = текущий бут (дов рестартовой части интервала — как у
    server_state, ограничено частотой тика).
    """
    lock_f = None
    try:
        import fcntl
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_f = open(LOCK_PATH, "w")
        fcntl.flock(lock_f, fcntl.LOCK_EX)
    except (ImportError, OSError):
        lock_f = None  # Windows/отказ — работаем без блокировки
    try:
        if not _any(cur):
            total = _norm_cur(_load_json_dict(STATE_PATH).get("baseline"))
            last_saved = _norm_cur(_load_json_dict(STATE_PATH).get("last"))
            return _merge_total(total, last_saved), total
        st = _load_json_dict(STATE_PATH)
        base = _norm_cur(st.get("baseline"))
        last = _norm_cur(st.get("last"))
        boot = _boot_id()
        restarted = bool(
            (boot and st.get("boot_id") and boot != st["boot_id"])
            or (last["drafts"] > 0 and cur["drafts"] < last["drafts"])
        )
        if restarted and _any(last):
            base = _norm_cur({"drafts": base["drafts"] + last["drafts"],
                              "draft_tokens": base["draft_tokens"] + last["draft_tokens"],
                              "accepted": base["accepted"] + last["accepted"],
                              "draft_pos": _vec_add(base["draft_pos"], last["draft_pos"]),
                              "accepted_pos": _vec_add(base["accepted_pos"], last["accepted_pos"])})
        if not _any(last):
            delta = {"drafts": 0, "draft_tokens": 0, "accepted": 0,
                     "draft_pos": [], "accepted_pos": []}  # первый запуск
        elif restarted:
            delta = dict(cur)  # счётчики сброшены: дельта интервала = новый бут
        else:
            delta = {"drafts": max(0, cur["drafts"] - last["drafts"]),
                     "draft_tokens": max(0, cur["draft_tokens"] - last["draft_tokens"]),
                     "accepted": max(0, cur["accepted"] - last["accepted"]),
                     "draft_pos": _vec_sub(cur["draft_pos"], last["draft_pos"]),
                     "accepted_pos": _vec_sub(cur["accepted_pos"], last["accepted_pos"])}
        _atomic_write_json(STATE_PATH, {"boot_id": boot, "last": cur,
                                        "baseline": base,
                                        "updated_at": _now_iso()})
        if _any(delta):
            _update_buckets(delta)
        total = _norm_cur({"drafts": cur["drafts"] + base["drafts"],
                           "draft_tokens": cur["draft_tokens"] + base["draft_tokens"],
                           "accepted": cur["accepted"] + base["accepted"],
                           "draft_pos": _vec_add(cur["draft_pos"], base["draft_pos"]),
                           "accepted_pos": _vec_add(cur["accepted_pos"], base["accepted_pos"])})
        return total, base
    finally:
        if lock_f:
            try:
                lock_f.close()
            except OSError:
                pass


def _merge_total(base: dict, last: dict) -> dict:
    """Итог для пассивного прогона без записи (движок поднялся, серий нет)."""
    return {"drafts": base["drafts"] + last["drafts"],
            "draft_tokens": base["draft_tokens"] + last["draft_tokens"],
            "accepted": base["accepted"] + last["accepted"],
            "draft_pos": _vec_add(base["draft_pos"], last["draft_pos"]),
            "accepted_pos": _vec_add(base["accepted_pos"], last["accepted_pos"])}


def _update_buckets(delta: dict) -> None:
    """Аддитивная дельта в почасовой бакет (UTC-час снимка); ничего не чистим."""
    buckets = _load_json_dict(BUCKETS_PATH)
    now = datetime.now(timezone.utc)
    day = buckets.setdefault(now.strftime("%Y-%m-%d"), {})
    key = now.strftime("%H:00-utc")
    entry = day.get(key)
    entry = entry if isinstance(entry, dict) else {}
    entry["drafts"] = _i(entry.get("drafts")) + delta["drafts"]
    entry["accepted"] = _i(entry.get("accepted")) + delta["accepted"]
    entry["draft_pos"] = _vec_add(entry.get("draft_pos") or [], delta["draft_pos"])
    entry["accepted_pos"] = _vec_add(entry.get("accepted_pos") or [], delta["accepted_pos"])
    day[key] = entry
    _atomic_write_json(BUCKETS_PATH, buckets)


def _pct(num: float | None, den: float | None, nd: int = 1) -> float | None:
    if not den or num is None:
        return None
    return round(num / den * 100.0, nd)


def _days_from_buckets(buckets: dict) -> list:
    """[{date, accept_rate, accept_len}] по возрастанию UTC-дат бакетов.

    accept_rate по дням — из позиционных сумм (sum accepted_pos / sum
    draft_pos), что равно accepted/draft_tokens, когда серии согласованы."""
    out = []
    for date in sorted(buckets):
        day = buckets.get(date)
        if not isinstance(day, dict):
            continue
        drafts = accepted = dpos = apos = 0
        for e in day.values():
            if not isinstance(e, dict):
                continue
            drafts += _i(e.get("drafts"))
            accepted += _i(e.get("accepted"))
            dpos += sum(_i(x) for x in (e.get("draft_pos") or []))
            apos += sum(_i(x) for x in (e.get("accepted_pos") or []))
        if not (drafts or dpos):
            continue
        out.append({"date": date,
                    "accept_rate": _pct(apos, dpos),
                    "accept_len": round(accepted / drafts, 2) if drafts else None})
    return out


def collect(days: float = 30.0) -> dict:
    """Собрать секцию spec. days не используется — сигнатура общая (CONTRACT)."""
    counters, pos = _snapshot()
    online = counters is not None
    total: dict | None = None
    if online and (counters or pos):
        try:
            total, _base = _update(_norm_cur({
                "drafts": counters.get("drafts", 0),
                "draft_tokens": counters.get("draft_tokens", 0),
                "accepted": counters.get("accepted", 0),
                "draft_pos": pos.get("draft_pos", []),
                "accepted_pos": pos.get("accepted_pos", []),
            }))
        except Exception:
            total = None  # со state-файлом проблемы — показываем как минимум бакеты
    if total is None:
        # нет свежих серий (движок молчал/спек-декод выключен) — итог из state,
        # чтобы rates не обнулялись на одном неуспешном тике
        st = _load_json_dict(STATE_PATH)
        total = _merge_total(_norm_cur(st.get("baseline")),
                             _norm_cur(st.get("last")))
    pos_rates = []
    for a, d in zip(total["accepted_pos"], total["draft_pos"]):
        pos_rates.append(_pct(a, d))
    return {
        "updated_at": _now_iso(),
        "online": online,
        "accept_rate": _pct(total["accepted"], total["draft_tokens"]),
        "accept_len": (round(total["accepted"] / total["drafts"], 2)
                       if total["drafts"] else None),
        "pos_rates": pos_rates,
        "counters": {k: total[k] for k in ("drafts", "draft_tokens", "accepted",
                                           "draft_pos", "accepted_pos")},
        "days": _days_from_buckets(_load_json_dict(BUCKETS_PATH)),
    }


def _fmt(v, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def main() -> int:
    try:  # Windows-консоль может быть не в UTF-8 — печать не должна ронять коллектор
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    snap = collect()
    out_dir = HERE / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "collect_spec.json"
    out_path.write_text(
        json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"spec online={snap['online']} accept_rate={_fmt(snap['accept_rate'])}% "
        f"accept_len={_fmt(snap['accept_len'], 2)} "
        f"pos_rates={[None if r is None else round(r, 1) for r in snap['pos_rates']]} "
        f"days={len(snap['days'])} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
