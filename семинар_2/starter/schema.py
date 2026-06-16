from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Воронеж",
]

CITY_DISTRICTS = {
    "Москва": ["ЦАО", "САО", "ЮАО", "ЗАО", "ВАО"],
    "Санкт-Петербург": ["Центральный", "Московский", "Приморский", "Калининский"],
    "Новосибирск": ["Центральный", "Октябрьский", "Ленинский", "Заельцовский"],
    "Екатеринбург": ["Ленинский", "Кировский", "Верх-Исетский", "Чкаловский"],
    "Казань": ["Вахитовский", "Советский", "Приволжский", "Кировский"],
    "Нижний Новгород": ["Нижегородский", "Советский", "Приокский", "Канавинский"],
    "Самара": ["Ленинский", "Самарский", "Промышленный", "Октябрьский"],
    "Краснодар": ["Центральный", "Западный", "Прикубанский", "Карасунский"],
    "Ростов-на-Дону": ["Кировский", "Ленинский", "Советский", "Ворошиловский"],
    "Воронеж": ["Центральный", "Ленинский", "Советский", "Коминтерновский"],
}

SPECIALITIES = [
    "врач-терапевт",
    "медицинская сестра",
    "учитель математики",
    "учитель русского языка",
    "инженер-проектировщик",
    "бухгалтер",
    "HR-специалист",
    "юрист",
    "специалист по охране труда",
    "IT-администратор",
]

DESIRED_COURSES = [
    "цифровые технологии в профессиональной деятельности",
    "управление персоналом",
    "охрана труда и техника безопасности",
    "проектное управление",
    "аналитика данных",
    "современные педагогические технологии",
    "медицинская документация и стандарты качества",
    "бухгалтерский учет и налоговое планирование",
]

City = Literal[
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Воронеж",
]

Speciality = Literal[
    "врач-терапевт",
    "медицинская сестра",
    "учитель математики",
    "учитель русского языка",
    "инженер-проектировщик",
    "бухгалтер",
    "HR-специалист",
    "юрист",
    "специалист по охране труда",
    "IT-администратор",
]

DesiredCourse = Literal[
    "цифровые технологии в профессиональной деятельности",
    "управление персоналом",
    "охрана труда и техника безопасности",
    "проектное управление",
    "аналитика данных",
    "современные педагогические технологии",
    "медицинская документация и стандарты качества",
    "бухгалтерский учет и налоговое планирование",
]


class Address(BaseModel):
    city: City
    district: str = Field(min_length=2, max_length=60)

    @model_validator(mode="after")
    def validate_district(self) -> "Address":
        allowed = CITY_DISTRICTS[self.city]

        if self.district not in allowed:
            raise ValueError(
                f"Район {self.district!r} не относится к городу {self.city!r}. "
                f"Допустимые районы: {allowed}"
            )

        return self


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=100)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        parts = value.strip().split()

        if len(parts) < 2:
            raise ValueError("ФИО должно содержать минимум имя и фамилию")

        if any(len(part) < 2 for part in parts):
            raise ValueError("Каждая часть ФИО должна быть длиннее одного символа")

        return " ".join(parts)

    @model_validator(mode="after")
    def validate_age_graduation_and_experience(self) -> "Application":
        current_year = date.today().year
        birth_year = current_year - self.age
        age_at_graduation = self.graduation_year - birth_year

        if age_at_graduation < 18:
            raise ValueError(
                "Возраст и год окончания противоречат друг другу: "
                "заявитель не мог окончить вуз раньше 18 лет"
            )

        if age_at_graduation > 45:
            raise ValueError(
                "Возраст и год окончания выглядят нереалистично: "
                "слишком позднее окончание вуза для указанного возраста"
            )

        if self.years_of_experience > max(0, self.age - 18):
            raise ValueError("Стаж не может быть больше возраста минус 18 лет")

        if self.graduation_year > current_year:
            raise ValueError("Год окончания не может быть в будущем")

        return self