import json

import pytest

from judgebench.judge.parser import extract_json_object


def test_extract_json_object_handles_code_fences() -> None:
    obj = extract_json_object("```json\n{\"a\": 1}\n```")
    assert obj == {"a": 1}


def test_extract_json_object_recovers_from_extra_text() -> None:
    obj = extract_json_object("Here you go:\n\n{\"a\": 1, \"b\": 2}\nThanks!")
    assert obj["b"] == 2


def test_extract_json_object_raises_on_empty() -> None:
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("   ")

