"""
Eval мульти-агента: 3 вопроса, на которых одиночный агент С5 ломается.

Каждый вопрос прогоняется дважды:
  1) через одиночного агента С5 (agent_s5.run_agent)
  2) через PWC-цикл (orchestrator.run_pwc)

и сравниваются:
  - вызван ли calculate там, где нужно (для арифметических вопросов)
  - нет ли галлюцинаций инструментов
  - есть ли в ответе обязательная подстрока (must_have)

Прогон N=5 раз, считаем долю успешных прогонов. Результат пишется в eval_pwc_results.json.

Запуск:
    python eval_pwc.py           # полный прогон
    python eval_pwc.py --single  # только один прогон каждого, быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; PWC обычно тоже (Планировщик может добавить "
            "выдуманный инструмент в план). Это — повод для Schema-Validator в домашке."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": [],
        "forbid_hallucinated_tools": True,
        "allow_honest_refusal": True,
    },
    {
        "id": "Q4",
        "query": "Что сейчас выше: ключевая ставка или индекс нищеты (инфляция плюс безработица)?",
        "comment": (
            "Кейс для валидатора: доступного get_unemployment в С6 нет, поэтому "
            "PWC без валидатора часто планирует выдуманный инструмент, а валидатор "
            "должен добиться честного отказа или перепланировки."
        ),
        "expected_tools_pwc": set(),
        "must_have_keywords": [],
        "forbid_hallucinated_tools": True,
        "allow_honest_refusal": True,
    },
    {
        "id": "Q5",
        "query": "Дай текущие курсы USD, EUR и CNY к рублю и кратко сравни их.",
        "comment": (
            "Естественная параллельность: три независимых валютных подвопроса "
            "можно исполнять одним уровнем."
        ),
        "expected_tools_pwc": {"get_fx_rate"},
        "must_have_keywords": ["USD", "EUR", "CNY"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q6",
        "query": "Насколько ключевая ставка выше инфляции за март 2026 года?",
        "comment": (
            "Реальный макро-вопрос: нужен номинальный показатель, инфляция "
            "и арифметика через calculate."
        ),
        "expected_tools_pwc": {"get_key_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = "calculate" in case["expected_tools_pwc"] and "calculate" not in used and bool(ans)
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    runtime_failure = "ratelimiterror" in ans or "planner failed" in ans
    honest_refusal = bool(
        case.get("allow_honest_refusal")
        and result.get("answer")
        and not runtime_failure
        and (
            "нельзя" in ans
            or "не решить" in ans
            or "недоступ" in ans
            or result.get("validator_errors")
        )
    )
    expected = set(case.get("expected_tools_pwc") or set())
    tools_ok = expected.issubset(used) or honest_refusal
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
        and tools_ok
        and not runtime_failure
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "expected_tools_ok": tools_ok,
        "honest_refusal": honest_refusal,
        "runtime_failure": runtime_failure,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc = {"runs": [], "pass": 0}
    pwc_validator = {"runs": [], "pass": 0}

    for i in range(n):
        # --- Одиночный агент ---
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC без валидатора ---
        try:
            r2 = run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                use_validator=False,
                parallel=True,
            )
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check2 = _check_pwc(case, r2)
        pwc["runs"].append(check2)
        pwc["pass"] += int(check2["ok"])

        # --- PWC + валидатор ---
        try:
            r3 = run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                use_validator=True,
                parallel=True,
            )
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check3 = _check_pwc(case, r3)
        pwc_validator["runs"].append(check3)
        pwc_validator["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_validator,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Только один прогон каждого кейса (быстро)")
    ap.add_argument("-n", type=int, default=5,
                    help="Сколько прогонов на кейс (default=5)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов × {n} прогонов\n")
    out = Path(__file__).parent / "eval_pwc_results.json"
    results = []
    done_ids: set[str] = set()
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(existing, list) and all(r.get("n") == n for r in existing):
                results = existing
                done_ids = {r.get("id") for r in existing}
                if done_ids:
                    print(f"[resume] уже есть: {', '.join(sorted(done_ids))}\n")
        except Exception:
            results = []
            done_ids = set()

    for case in CASES:
        if case["id"] in done_ids:
            print(f"=== {case['id']}: пропуск, уже сохранено")
            continue
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        s = r["single"]; p = r["pwc"]; pv = r["pwc_validator"]
        print(f"   single: {s['pass']}/{n}    pwc: {p['pass']}/{n}    pwc+validator: {pv['pass']}/{n}")
        for run in p["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ План содержит выдуманные инструменты: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 60)
    print("ИТОГО:")
    for r in results:
        print(f"  {r['id']}: single {r['single']['pass']}/{n}  "
              f"pwc {r['pwc']['pass']}/{n}  "
              f"pwc+validator {r['pwc_validator']['pass']}/{n}  — {r['query'][:60]}")

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
