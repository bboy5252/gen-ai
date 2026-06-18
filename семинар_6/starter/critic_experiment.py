from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("LLM_TIMEOUT", "45")

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer


def _sq(id_: int, question: str, tools: list[str], deps: list[int] | None = None) -> SubQuestion:
    return SubQuestion(
        id=id_,
        question=question,
        expected_tools=tools,
        depends_on=deps or [],
    )


def _wa(id_: int, answer: str, tools: list[str]) -> WorkerAnswer:
    return WorkerAnswer(
        subquestion_id=id_,
        question_snippet=f"fake {id_}",
        answer=answer,
        used_tools=tools,
    )


FAKE_BROKEN = [
    {
        "label": "арифметика без calculate",
        "question": "Какая разница между курсами USD и EUR?",
        "plan": Plan(
            reasoning="Нужны два курса и разность.",
            subquestions=[
                _sq(1, "Курс USD сегодня?", ["get_fx_rate"]),
                _sq(2, "Курс EUR сегодня?", ["get_fx_rate"]),
                _sq(3, "Разница EUR и USD?", ["calculate"], [1, 2]),
            ],
        ),
        "answers": {
            1: _wa(1, "USD=82.5 руб.", ["get_fx_rate"]),
            2: _wa(2, "EUR=89.0 руб.", ["get_fx_rate"]),
            3: _wa(3, "Разница 6.5 руб.", []),
        },
    },
    {
        "label": "выдуманное число",
        "question": "Какая сейчас ключевая ставка?",
        "plan": Plan(
            reasoning="Нужно получить ключевую ставку.",
            subquestions=[_sq(1, "Ключевая ставка сегодня?", ["get_key_rate"])],
        ),
        "answers": {1: _wa(1, "Ключевая ставка 99% годовых.", ["get_key_rate"])},
    },
    {
        "label": "несогласованные данные",
        "question": "Во сколько раз USD вырос с 2022 года?",
        "plan": Plan(
            reasoning="Нужны два курса и отношение.",
            subquestions=[
                _sq(1, "USD на 2022-01-01?", ["get_fx_rate"]),
                _sq(2, "USD сегодня?", ["get_fx_rate"]),
                _sq(3, "Отношение текущего курса к старому?", ["calculate"], [1, 2]),
            ],
        ),
        "answers": {
            1: _wa(1, "USD=75 руб.", ["get_fx_rate"]),
            2: _wa(2, "USD=90 руб.", ["get_fx_rate"]),
            3: _wa(3, "Курс вырос в 3 раза.", ["calculate"]),
        },
    },
    {
        "label": "ошибка исполнителя",
        "question": "Какая инфляция за март 2026?",
        "plan": Plan(
            reasoning="Нужен ИПЦ за месяц.",
            subquestions=[_sq(1, "ИПЦ за март 2026?", ["get_inflation"])],
        ),
        "answers": {1: _wa(1, "(ошибка: tool call failed)", ["get_inflation"])},
    },
    {
        "label": "план не покрывает вопрос",
        "question": "Какая сейчас реальная ключевая ставка?",
        "plan": Plan(
            reasoning="Получим только номинальную ставку.",
            subquestions=[_sq(1, "Ключевая ставка сегодня?", ["get_key_rate"])],
        ),
        "answers": {1: _wa(1, "Ключевая ставка 16% годовых.", ["get_key_rate"])},
    },
]


def run_experiment(n: int) -> list[dict]:
    rows: list[dict] = []
    for case in FAKE_BROKEN:
        row = {"case": case["label"], "temperature_0_0_false_accepts": 0, "temperature_0_7_false_accepts": 0, "errors": []}
        print(f"[case] {case['label']}", flush=True)
        for temperature, key in [(0.0, "temperature_0_0_false_accepts"), (0.7, "temperature_0_7_false_accepts")]:
            for _ in range(n):
                try:
                    verdict = critic(
                        case["question"],
                        case["plan"],
                        case["answers"],
                        temperature=temperature,
                        max_retries=0,
                    )
                    if verdict.ok:
                        row[key] += 1
                except Exception as e:
                    row["errors"].append(f"T={temperature}: {type(e).__name__}: {e}")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    rows = run_experiment(args.n)
    print("| Битый кейс | T=0.0, ложных принятий | T=0.7, ложных принятий | Ошибок API |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['case']} | {row['temperature_0_0_false_accepts']}/{args.n} "
            f"| {row['temperature_0_7_false_accepts']}/{args.n} "
            f"| {len(row.get('errors', []))} |"
        )

    out = Path(__file__).parent / "critic_experiment_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
