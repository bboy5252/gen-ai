"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана. Пустой список означает валидный план."""
    errors: list[str] = []
    seen_ids: set[int] = set()

    for sq in plan.subquestions:
        if sq.id in seen_ids:
            errors.append(f"дублируется id подвопроса {sq.id}")
        seen_ids.add(sq.id)

        for tool in sq.expected_tools:
            if not isinstance(tool, str):
                errors.append(f"подвопрос {sq.id}: tool должен быть строкой")
                continue
            name = tool.strip()
            if name != tool or "(" in name or ")" in name or "," in name:
                errors.append(
                    f"подвопрос {sq.id}: expected_tools должен содержать только имя, не вызов: {tool}"
                )
                continue
            if name not in VALID_TOOLS:
                errors.append(f"подвопрос {sq.id}: неизвестный инструмент {name}")

        for dep_id in sq.depends_on:
            if dep_id == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит сам от себя")

    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни: внутри уровня нет зависимостей друг от друга."""
    by_id = {s.id: s for s in subqs}
    visiting: set[int] = set()
    done: dict[int, int] = {}

    def depth(node_id: int, path: list[int]) -> int:
        if node_id not in by_id:
            return -1
        if node_id in done:
            return done[node_id]
        if node_id in visiting:
            cycle = " -> ".join(map(str, path + [node_id]))
            raise ValueError(f"цикл в depends_on: {cycle}")

        visiting.add(node_id)
        dep_depths = [
            depth(dep, path + [node_id])
            for dep in by_id[node_id].depends_on
            if dep in by_id
        ]
        visiting.remove(node_id)
        done[node_id] = (max(dep_depths) + 1) if dep_depths else 0
        return done[node_id]

    for sq in subqs:
        depth(sq.id, [])

    levels: dict[int, list[SubQuestion]] = {}
    for sq in subqs:
        levels.setdefault(done[sq.id], []).append(sq)
    return [levels[i] for i in sorted(levels)]


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    return [sq for level in _topological_levels(subqs) for sq in level]


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы одного уровня."""
    if not level:
        return {}
    if not parallel or len(level) == 1:
        return {sq.id: worker(sq, prev_answers=prev_answers) for sq in level}

    out: dict[int, WorkerAnswer] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(level))) as pool:
        future_to_sq = {
            pool.submit(worker, sq, prev_answers): sq
            for sq in level
        }
        for fut in as_completed(future_to_sq):
            sq = future_to_sq[fut]
            try:
                out[sq.id] = fut.result()
            except Exception as e:
                out[sq.id] = WorkerAnswer(
                    subquestion_id=sq.id,
                    question_snippet=sq.question[:60],
                    answer=f"(ошибка: {type(e).__name__}: {e})",
                    used_tools=[],
                    raw_trace=[],
                )
    return out


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [
        f"{i}. {answers[i].answer}"
        for i in sorted(answers)
    ]
    prompt = (
        "Собери финальный ответ пользователю в 1-2 фразы. "
        "Не добавляй новых чисел, используй только ответы ниже.\n\n"
        f"Вопрос: {question}\n\n"
        f"Ответы исполнителей:\n" + "\n".join(parts)
    )
    try:
        client = make_raw_client()
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": "Ты аккуратно синтезируешь краткий ответ."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or " · ".join(parts)
    except Exception:
        return " · ".join(parts)


def _append_plan_trace(trace: list[dict[str, Any]], plan: Plan, *, iter_num: int, kind: str) -> None:
    trace.append(
        {
            "iter": iter_num,
            "kind": kind,
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )


def _plan_with_validation(
    question: str,
    trace: list[dict[str, Any]],
    *,
    iter_num: int,
    feedback: str | None = None,
    use_validator: bool = True,
) -> tuple[Plan, list[str]]:
    try:
        plan = planner(question, feedback=feedback)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        trace.append({"iter": iter_num, "kind": "planner_error", "error": error})
        return Plan(reasoning=f"planner failed: {error}", subquestions=[]), [error]
    _append_plan_trace(trace, plan, iter_num=iter_num, kind="plan")
    if not use_validator:
        return plan, []

    errors = validate_plan(plan)
    trace.append({"iter": iter_num, "kind": "validator", "errors": errors})
    if not errors:
        return plan, []

    try:
        plan = planner(
            question,
            feedback=(
                "Инструменты не существуют или записаны неверно: "
                + "; ".join(errors)
                + ". Используй только get_fx_rate, get_key_rate, get_inflation, calculate. "
                + "Если задача не решается этими tools, верни пустой план."
            ),
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        trace.append({"iter": iter_num, "kind": "planner_error_after_validator", "error": error})
        return Plan(reasoning=f"planner failed: {error}", subquestions=[]), errors + [error]
    _append_plan_trace(trace, plan, iter_num=iter_num, kind="plan_after_validator")
    errors = validate_plan(plan)
    trace.append({"iter": iter_num, "kind": "validator_after_feedback", "errors": errors})
    return plan, errors


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan, validator_errors = _plan_with_validation(
        question,
        trace,
        iter_num=0,
        use_validator=use_validator,
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    if not plan.subquestions:
        answer = f"Задачу нельзя корректно решить доступными инструментами: {plan.reasoning}"
        return {
            "answer": answer,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
            "validator_errors": validator_errors,
        }

    if validator_errors:
        answer = (
            "Задачу нельзя корректно решить доступными инструментами: "
            + "; ".join(validator_errors)
        )
        return {
            "answer": answer,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
            "validator_errors": validator_errors,
        }

    answers: dict[int, WorkerAnswer] = {}
    for iter_num in range(1, max_iter + 1):
        answers = {}
        levels = _topological_levels(plan.subquestions)
        for level_idx, level in enumerate(levels, start=1):
            level_answers = execute_level(level, answers, parallel=parallel)
            for sq in level:
                ans = level_answers[sq.id]
                answers[sq.id] = ans
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "level": level_idx,
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            feedback = verdict.reason
        else:
            feedback = f"{verdict.reason}. Переделай подвопросы: {verdict.rework_ids}"

        plan, validator_errors = _plan_with_validation(
            question,
            trace,
            iter_num=iter_num,
            feedback=feedback,
            use_validator=use_validator,
        )
        if not plan.subquestions or validator_errors:
            answer = (
                "Задачу нельзя корректно решить доступными инструментами: "
                + (plan.reasoning if not validator_errors else "; ".join(validator_errors))
            )
            return {
                "answer": answer,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
                "validator_errors": validator_errors,
            }

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
