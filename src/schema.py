"""Строгая схема chat-примера и потоковая валидация JSONL."""

import json
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

ROLES = ("system", "user", "assistant")


class SchemaError(ValueError):
    """Ошибка схемы с именем файла и номером строки."""


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content пустой")
        return value


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    topic: str
    messages: list[Message]

    @model_validator(mode="after")
    def exact_roles(self) -> "Example":
        got = tuple(message.role for message in self.messages)
        if got != ROLES:
            raise ValueError(f"роли должны идти ровно как {ROLES}, получено {got or '()'}")
        if not self.id.strip() or not self.topic.strip():
            raise ValueError("id и topic не могут быть пустыми")
        return self

    @property
    def user(self) -> str:
        return self.messages[1].content

    @property
    def assistant(self) -> str:
        return self.messages[2].content


def explain(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<корень>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def iter_examples(path: str | Path) -> Iterator[Example]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                raise SchemaError(f"{path}:{line_number}: пустая строка в JSONL")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_number}: не разбирается как JSON — {exc.msg}") from exc
            try:
                yield Example.model_validate(payload)
            except ValidationError as exc:
                raise SchemaError(f"{path}:{line_number}: {explain(exc)}") from exc


def dump(example: Example) -> str:
    return json.dumps(example.model_dump(), ensure_ascii=False)
