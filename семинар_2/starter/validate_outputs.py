from __future__ import annotations

import sys

import pandas as pd

from schema import Application


def validate_row(row: dict) -> Application:
    return Application(
        full_name=row["full_name"],
        age=int(row["age"]),
        address={
            "city": row["city"],
            "district": row["district"],
        },
        speciality=row["speciality"],
        desired_course=row["desired_course"],
        years_of_experience=int(row["years_of_experience"]),
        graduation_year=int(row["graduation_year"]),
    )


def main(path: str = "applications.csv") -> None:
    df = pd.read_csv(path)

    valid = []

    for i, row in df.iterrows():
        try:
            validate_row(row.to_dict())
            valid.append(True)
        except Exception as e:
            valid.append(False)
            print(f"Ошибка в строке {i + 1}: {e}")

    city_top = df["city"].value_counts(normalize=True).max()
    speciality_top = df["speciality"].value_counts(normalize=True).max()

    print(f"Валидных заявок: {sum(valid)}/{len(df)}")
    print(f"Максимальная доля города: {city_top:.0%}")
    print(f"Максимальная доля специальности: {speciality_top:.0%}")

    assert len(df) == 50, f"Должно быть 50 строк, получилось {len(df)}"
    assert all(valid), "Есть невалидные заявки"
    assert city_top <= 0.40, "Превышен порог 40% по городам"
    assert speciality_top <= 0.35, "Превышен порог 35% по специальностям"

    print("OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "applications.csv")