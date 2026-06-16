from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from llm_client import get_model, make_client
from schema import Application


SYSTEM_PROMPT = """
Ты генерируешь тестовую заявку на курс ДПО.
Верни только JSON.
""".strip()


CONFLICT_PROMPT = """
Создай одну заявку на курс ДПО.

Конфликтное условие:
- desired_course должен быть оригинальным авторским курсом, которого нет в стандартных списках.
- При этом все поля должны пройти строгую Pydantic-схему Application.

Город: Москва.
Район: ЦАО.
Возраст: 35.
Год окончания: 2012.
Стаж: 10.
Специальность: юрист.
""".strip()


def run_stress_test(max_retries: int) -> bool:
    client = make_client()

    try:
        app = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CONFLICT_PROMPT},
            ],
            response_model=Application,
            max_retries=max_retries,
            temperature=0.8,
        )

        print(f"max_retries={max_retries}: модель прошла валидацию")
        print(app.model_dump())
        return True

    except Exception as e:
        print(f"max_retries={max_retries}: модель не прошла валидацию")
        print(e)
        return False


def main() -> None:
    for retries in [0, 1, 2, 3]:
        run_stress_test(retries)


if __name__ == "__main__":
    main()