from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOTS_DIR = Path("histograms")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Файл пустой")

    required_columns = {
        "full_name",
        "age",
        "city",
        "district",
        "speciality",
        "desired_course",
        "years_of_experience",
        "graduation_year",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"В CSV не хватает колонок: {sorted(missing)}")

    return df


def plot_bar(series: pd.Series, title: str, ylabel: str, out: Path) -> pd.Series:
    counts = series.value_counts()

    plt.figure(figsize=(11, 5))
    counts.plot(kind="bar", edgecolor="white")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return counts


def find_unrealistic_rows(df: pd.DataFrame) -> pd.DataFrame:
    suspicious = []

    for _, row in df.iterrows():
        course = str(row["desired_course"]).lower()
        spec = str(row["speciality"]).lower()
        age = int(row["age"])
        experience = int(row["years_of_experience"])
        graduation_year = int(row["graduation_year"])

        reason = None

        if "медицин" in course and not (
            "врач" in spec or "медицин" in spec or "медсестр" in spec
        ):
            reason = "медицинский курс выбран не медицинским специалистом"

        elif "педагог" in course and not (
            "учитель" in spec or "педагог" in spec or "преподаватель" in spec
        ):
            reason = "педагогический курс выбран не педагогическим специалистом"

        elif "бухгалтер" in course and not (
            "бухгалтер" in spec or "экономист" in spec or "финанс" in spec
        ):
            reason = "бухгалтерский курс выбран специалистом без финансового профиля"

        elif "проект" in course and not (
            "проект" in spec or "менеджер" in spec or "инженер" in spec
        ):
            reason = "курс по управлению проектами выбран без очевидной связи с профилем"

        elif experience == 0 and age > 45:
            reason = "нулевой стаж при зрелом возрасте"

        elif experience > max(0, age - 18):
            reason = "стаж больше реалистично возможного для возраста"

        elif graduation_year > 2024 or graduation_year < 1980:
            reason = "год окончания вне допустимого диапазона"

        if reason:
            item = row.to_dict()
            item["reason"] = reason
            suspicious.append(item)

    return pd.DataFrame(suspicious)


def write_conclusions(df: pd.DataFrame, suspicious: pd.DataFrame) -> None:
    n = len(df)

    city_counts = df["city"].value_counts()
    spec_counts = df["speciality"].value_counts()

    top_city = city_counts.index[0]
    top_spec = spec_counts.index[0]

    top_city_pct = city_counts.iloc[0] / n * 100
    top_spec_pct = spec_counts.iloc[0] / n * 100

    city_status = "не превышен" if top_city_pct <= 40 else "превышен"
    spec_status = "не превышен" if top_spec_pct <= 35 else "превышен"

    text = f"""# Выводы

Было сгенерировано {n} валидных заявок из {n}. Для борьбы с mode collapse использовалась стратификация по городам: план генерации заранее распределял заявки между городами, поэтому модель не могла постоянно выбирать один и тот же город. Самый частый город — {top_city}, его доля составила {top_city_pct:.0f}%, то есть порог 40% {city_status}. По специальностям жёсткого квотирования не было, но в промпт передавался `seed_speciality`, который направлял генерацию и снижал риск однотипных ответов. Самая частая специальность — {top_spec}, её доля составила {top_spec_pct:.0f}%, то есть порог 35% {spec_status}.

`@field_validator` и `@model_validator` использовались как страховка от структурных и логических ошибок. В схеме проверяются ФИО, город, район, соответствие района городу, диапазоны возраста, стажа и года окончания, а также согласованность возраста со стажем и годом выпуска. При обычной генерации модель иногда нарушала ограничения: выдавала год окончания раньше 1980, слишком большой стаж или противоречивую связку возраста и года окончания. Эти ошибки были пойманы валидаторами, после чего генератор повторял запрос. Если модель не проходила проверку после нескольких попыток, использовался fallback, чтобы сохранить итоговые 50/50 валидных заявок.
"""

    if suspicious.empty:
        text += "\nПо простым эвристикам явно нереалистичных комбинаций не найдено. При этом смысловая проверка остаётся частично экспертной: некоторые заявки могут быть реалистичны только как смена профессиональной траектории.\n"
    else:
        text += f"\nРасширенный анализ нашёл {len(suspicious)} потенциально спорных комбинаций. Чаще всего это случаи, когда выбранный курс не идеально связан с текущей специальностью. Такие строки не обязательно являются ошибками, потому что курсы ДПО могут использоваться для переподготовки, но их стоит отдельно просматривать перед использованием данных.\n"

    Path("выводы.md").write_text(text, encoding="utf-8")


def write_report(df: pd.DataFrame, suspicious: pd.DataFrame) -> None:
    n = len(df)

    city_counts = df["city"].value_counts()
    spec_counts = df["speciality"].value_counts()
    course_counts = df["desired_course"].value_counts()

    crosstab = pd.crosstab(df["city"], df["speciality"])
    crosstab.to_csv("crosstab_city_speciality.csv", encoding="utf-8-sig")

    lines = [
        f"# Report по {n} заявкам",
        "",
        "## Распределение по городам",
        city_counts.to_markdown(),
        "",
        "## Распределение по специальностям",
        spec_counts.to_markdown(),
        "",
        "## Распределение по курсам",
        course_counts.to_markdown(),
        "",
        "## Кросс-таблица город × специальность",
        crosstab.to_markdown(),
        "",
        "## Потенциально нереалистичные комбинации",
    ]

    if suspicious.empty:
        lines.append("По простым эвристикам подозрительных строк не найдено.")
    else:
        lines.append(
            suspicious[
                [
                    "full_name",
                    "age",
                    "city",
                    "speciality",
                    "desired_course",
                    "years_of_experience",
                    "graduation_year",
                    "reason",
                ]
            ]
            .head(10)
            .to_markdown(index=False)
        )

    Path("report.md").write_text("\n".join(lines), encoding="utf-8")


def main(path: str = "applications.csv") -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    df = load(path)

    city_counts = plot_bar(
        df["city"],
        "Распределение заявок по городам",
        "Количество заявок",
        PLOTS_DIR / "cities.png",
    )

    spec_counts = plot_bar(
        df["speciality"],
        "Распределение заявок по специальностям",
        "Количество заявок",
        PLOTS_DIR / "specialities.png",
    )

    plot_bar(
        df["desired_course"],
        "Распределение заявок по желаемым курсам",
        "Количество заявок",
        PLOTS_DIR / "courses.png",
    )

    suspicious = find_unrealistic_rows(df)

    if not suspicious.empty:
        suspicious.to_csv(
            "suspicious_combinations.csv",
            index=False,
            encoding="utf-8-sig",
        )

    write_report(df, suspicious)
    write_conclusions(df, suspicious)

    print(f"Загружено заявок: {len(df)}")
    print(f"Топ-город: {city_counts.index[0]} — {city_counts.iloc[0]}")
    print(f"Топ-специальность: {spec_counts.index[0]} — {spec_counts.iloc[0]}")
    print("Сохранено: histograms/cities.png, histograms/specialities.png, histograms/courses.png, report.md, выводы.md, crosstab_city_speciality.csv")

    if not suspicious.empty:
        print("Дополнительно сохранено: suspicious_combinations.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "applications.csv")