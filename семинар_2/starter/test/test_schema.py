from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from schema import Address, Application


def test_valid_application() -> None:
    app = Application(
        full_name="Иванова Елена Васильевна",
        age=45,
        address=Address(city="Нижний Новгород", district="Нижегородский"),
        speciality="медицинская сестра",
        desired_course="медицинская документация и стандарты качества",
        years_of_experience=20,
        graduation_year=2002,
    )

    assert app.age == 45
    assert app.address.city == "Нижний Новгород"
    assert app.graduation_year == 2002


def test_invalid_city() -> None:
    with pytest.raises(ValidationError):
        Address(city="Тула", district="Центральный")


def test_invalid_district_for_city() -> None:
    with pytest.raises(ValidationError):
        Address(city="Москва", district="Нижегородский")


def test_invalid_speciality() -> None:
    with pytest.raises(ValidationError):
        Application(
            full_name="Петров Петр Петрович",
            age=35,
            address=Address(city="Москва", district="ЦАО"),
            speciality="архитектор",
            desired_course="проектное управление",
            years_of_experience=10,
            graduation_year=2012,
        )


def test_invalid_course() -> None:
    with pytest.raises(ValidationError):
        Application(
            full_name="Петров Петр Петрович",
            age=35,
            address=Address(city="Москва", district="ЦАО"),
            speciality="юрист",
            desired_course="оригинальный авторский курс",
            years_of_experience=10,
            graduation_year=2012,
        )


def test_invalid_graduation_year_too_early_for_age() -> None:
    with pytest.raises(ValidationError):
        Application(
            full_name="Сидорова Анна Павловна",
            age=25,
            address=Address(city="Казань", district="Вахитовский"),
            speciality="бухгалтер",
            desired_course="бухгалтерский учет и налоговое планирование",
            years_of_experience=3,
            graduation_year=1980,
        )


def test_invalid_experience_for_age() -> None:
    with pytest.raises(ValidationError):
        Application(
            full_name="Смирнов Алексей Сергеевич",
            age=24,
            address=Address(city="Самара", district="Ленинский"),
            speciality="IT-администратор",
            desired_course="аналитика данных",
            years_of_experience=20,
            graduation_year=2023,
        )


def test_invalid_full_name_too_short() -> None:
    with pytest.raises(ValidationError):
        Application(
            full_name="Иван",
            age=35,
            address=Address(city="Москва", district="ЦАО"),
            speciality="юрист",
            desired_course="цифровые технологии в профессиональной деятельности",
            years_of_experience=10,
            graduation_year=2012,
        )