from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from llm_client import get_model, make_client
from schema import Application, CITIES, CITY_DISTRICTS, DESIRED_COURSES, SPECIALITIES

N_APPLICATIONS = 50
MAX_RETRIES = 3
OUTER_RETRIES = 5
TEMPERATURE = 0.65
RANDOM_SEED = 42
OUT_JSON = "applications.json"
OUT_CSV = "applications.csv"

SYSTEM_PROMPT = """
Ты генерируешь синтетические заявки на курсы повышения квалификации ДПО.
Нужно создать реалистичного взрослого специалиста из России.

Правила:
- ответ должен быть только JSON;
- город должен быть только из разрешённого списка;
- район должен соответствовать выбранному городу;
- специальность и курс должны быть только из разрешённых списков;
- возраст, год окончания и стаж должны быть согласованы между собой;
- заявитель не может окончить вуз раньше 18 лет;
- стаж не может быть больше возраста минус 18;
- не повторяй одни и те же ФИО и не делай всех заявителей одинаковыми.
""".strip()


def build_user_prompt(seed_city: str, seed_speciality: str, seed_number: int) -> str:
    districts = ", ".join(CITY_DISTRICTS[seed_city])

    return f"""
Создай одну валидную заявку на курс ДПО.

seed_number: {seed_number}
seed_city: {seed_city}
seed_speciality: {seed_speciality}

Жёсткие ограничения:
- address.city должен быть ровно: {seed_city}
- address.district выбери только из списка для этого города: {districts}
- speciality желательно сделать близкой к seed_speciality: {seed_speciality}
- desired_course выбери из списка: {", ".join(DESIRED_COURSES)}
- graduation_year: 1980-2024
- age: 22-65
- years_of_experience: 0-40
- graduation_year должен быть реалистичен для возраста
- заявитель не может окончить вуз раньше 18 лет
- years_of_experience не может быть больше age - 18

Разрешённые города: {", ".join(CITIES)}
Разрешённые специальности: {", ".join(SPECIALITIES)}
""".strip()


def make_city_plan() -> list[str]:
    cities = []

    while len(cities) < N_APPLICATIONS:
        cities.extend(CITIES)

    cities = cities[:N_APPLICATIONS]
    random.Random(RANDOM_SEED).shuffle(cities)

    return cities


def generate_one(client, seed_city: str, seed_speciality: str, seed_number: int) -> Application:
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(seed_city, seed_speciality, seed_number),
            },
        ],
        response_model=Application,
        max_retries=MAX_RETRIES,
        temperature=TEMPERATURE,
    )


def make_fallback_application(
    seed_city: str,
    seed_speciality: str,
    seed_number: int,
) -> Application:
    first_names = [
        "Анна",
        "Мария",
        "Елена",
        "Ольга",
        "Ирина",
        "Алексей",
        "Дмитрий",
        "Сергей",
        "Павел",
        "Игорь",
        "Наталья",
        "Татьяна",
        "Виктория",
        "Андрей",
        "Михаил",
    ]

    last_names = [
        "Иванова",
        "Петрова",
        "Соколова",
        "Кузнецова",
        "Смирнова",
        "Волкова",
        "Морозова",
        "Новикова",
        "Федорова",
        "Попова",
        "Иванов",
        "Петров",
        "Соколов",
        "Кузнецов",
        "Смирнов",
    ]

    patronymics = [
        "Андреевна",
        "Сергеевна",
        "Дмитриевна",
        "Игоревна",
        "Павловна",
        "Александровна",
        "Викторовна",
        "Михайловна",
        "Андреевич",
        "Сергеевич",
        "Дмитриевич",
        "Игоревич",
        "Павлович",
        "Александрович",
        "Викторович",
        "Михайлович",
    ]

    course_by_speciality = {
        "врач-терапевт": "медицинская документация и стандарты качества",
        "медицинская сестра": "медицинская документация и стандарты качества",
        "учитель математики": "современные педагогические технологии",
        "учитель русского языка": "современные педагогические технологии",
        "инженер-проектировщик": "проектное управление",
        "бухгалтер": "бухгалтерский учет и налоговое планирование",
        "HR-специалист": "управление персоналом",
        "юрист": "цифровые технологии в профессиональной деятельности",
        "специалист по охране труда": "охрана труда и техника безопасности",
        "IT-администратор": "аналитика данных",
    }

    rng = random.Random(RANDOM_SEED + seed_number)

    age = rng.randint(24, 62)
    birth_year = 2026 - age

    min_graduation_year = max(1980, birth_year + 20)
    max_graduation_year = min(2024, birth_year + 28)

    if min_graduation_year > max_graduation_year:
        min_graduation_year = max(1980, birth_year + 18)
        max_graduation_year = min(2024, birth_year + 35)

    graduation_year = rng.randint(min_graduation_year, max_graduation_year)
    max_experience = min(40, age - 18, 2026 - graduation_year)
    years_of_experience = rng.randint(0, max(0, max_experience))

    return Application(
        full_name=f"{rng.choice(last_names)} {rng.choice(first_names)} {rng.choice(patronymics)}",
        age=age,
        address={
            "city": seed_city,
            "district": rng.choice(CITY_DISTRICTS[seed_city]),
        },
        speciality=seed_speciality,
        desired_course=course_by_speciality.get(
            seed_speciality,
            rng.choice(DESIRED_COURSES),
        ),
        years_of_experience=years_of_experience,
        graduation_year=graduation_year,
    )


def generate_valid_one(
    client,
    seed_city: str,
    seed_speciality: str,
    seed_number: int,
) -> Application:
    last_error = None

    for attempt in range(1, OUTER_RETRIES + 1):
        try:
            return generate_one(
                client=client,
                seed_city=seed_city,
                seed_speciality=seed_speciality,
                seed_number=seed_number * 100 + attempt,
            )
        except Exception as e:
            last_error = e
            print(f"Повтор {attempt}/{OUTER_RETRIES} для заявки {seed_number}: {e}")

    print(
        f"Fallback для заявки {seed_number}: LLM не смогла пройти валидацию. "
        f"Последняя ошибка: {last_error}"
    )

    return make_fallback_application(
        seed_city=seed_city,
        seed_speciality=seed_speciality,
        seed_number=seed_number,
    )


def flatten_application(app: Application) -> dict:
    row = app.model_dump()
    address = row.pop("address")
    row["city"] = address["city"]
    row["district"] = address["district"]

    return row


def save_outputs(applications: list[Application]) -> pd.DataFrame:
    json_data = [app.model_dump(mode="json") for app in applications]

    Path(OUT_JSON).write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df = pd.DataFrame([flatten_application(app) for app in applications])

    df = df[
        [
            "full_name",
            "age",
            "city",
            "district",
            "speciality",
            "desired_course",
            "years_of_experience",
            "graduation_year",
        ]
    ]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    return df


def validate_distribution(df: pd.DataFrame) -> None:
    city_top = df["city"].value_counts(normalize=True).max()
    speciality_top = df["speciality"].value_counts(normalize=True).max()

    assert len(df) == N_APPLICATIONS, (
        f"Должно быть {N_APPLICATIONS} заявок, получилось {len(df)}"
    )

    assert city_top <= 0.40, f"Перекос по городам: {city_top:.0%} > 40%"
    assert speciality_top <= 0.35, (
        f"Перекос по специальностям: {speciality_top:.0%} > 35%"
    )


def main() -> None:
    random.seed(RANDOM_SEED)

    client = make_client()
    city_plan = make_city_plan()
    applications: list[Application] = []

    for i, seed_city in enumerate(city_plan, start=1):
        seed_speciality = random.choice(SPECIALITIES)
        app = generate_valid_one(client, seed_city, seed_speciality, i)
        applications.append(app)

        print(
            f"[{i:02d}/{N_APPLICATIONS}] "
            f"{app.full_name} — "
            f"возраст: {app.age}, "
            f"город: {app.address.city}, "
            f"район: {app.address.district}, "
            f"специальность: {app.speciality}, "
            f"курс: {app.desired_course}, "
            f"стаж: {app.years_of_experience}, "
            f"год выпуска: {app.graduation_year}"
        )

    df = save_outputs(applications)
    validate_distribution(df)

    subprocess.run([sys.executable, "analysis_dpo.py", OUT_CSV], check=True)

    print("\nГотово: applications.csv, cities.png, specialities.png, report.md, выводы.md")


if __name__ == "__main__":
    main()