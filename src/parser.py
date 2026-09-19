"""Module for loading and validating input schemas and test prompts."""

import json
import os
from typing import Annotated, Any, List, Tuple

from pydantic import Field, TypeAdapter, ValidationError

from .models import FunctionDef, PromptDef


def _reject_duplicate_keys(ordered_pairs: List[Tuple[str, Any]]) -> dict:
    """Ensure no duplicate keys exist in a JSON object."""
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            raise ValueError(f"Duplicate JSON key found: {key}")
        d[key] = value
    return d


def _extract_data(filepath: str) -> str:
    """Validate filesystem status and read raw file contents.

    Args:
        filepath: Path to the target file.

    Returns:
        The raw string content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If read permissions are missing.
        ValueError: If the path is not a regular file or contains
            invalid UTF-8 encoding.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a regular file: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File {filepath} is not valid UTF-8: {exc}"
        ) from exc
    except PermissionError as exc:
        raise PermissionError(
            f"Insufficient permissions to read {filepath}: {exc}"
        ) from exc


def parse_function_definitions(filepath: str) -> List[FunctionDef]:
    """Load and validate function definitions from a JSON file.

    Args:
        filepath: The file path to the functions definition JSON file.

    Returns:
        A list of validated FunctionDef objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If read permissions are missing.
        ValueError: If the file contains invalid JSON, is empty,
            or fails schema validation.
    """
    raw_data = _extract_data(filepath)

    if not raw_data.strip():
        raise ValueError(f"File is empty: {filepath}")

    try:
        json.loads(raw_data, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid JSON in {filepath}: {exc}"
        ) from exc

    adapter: TypeAdapter[List[FunctionDef]] = TypeAdapter(
        Annotated[List[FunctionDef], Field(min_length=1)]
    )
    try:
        result: List[FunctionDef] = adapter.validate_json(raw_data)

        for f in result:
            if not f.name.isidentifier():
                raise ValueError(
                    f"Function name '{f.name}' is not a valid identifier"
                )
            for p_name in f.parameters.keys():
                if not p_name.isidentifier():
                    raise ValueError(
                        f"Parameter '{p_name}' in '{f.name}' "
                        "is not a valid identifier"
                    )

        names = [f.name for f in result]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate function names found in {filepath}")

        descriptions = [f.description for f in result]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError(
                f"Duplicate function descriptions found in {filepath}"
            )

        return result
    except ValidationError as exc:
        raise ValueError(
            f"Schema validation error in {filepath}: {exc}"
        ) from exc


def parse_prompts(filepath: str) -> List[PromptDef]:
    """Load and validate test prompts from a JSON file.

    Args:
        filepath: The file path to the test prompts JSON file.

    Returns:
        A list of validated PromptDef objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If read permissions are missing.
        ValueError: If the file contains invalid JSON, is empty,
            or fails schema validation.
    """
    raw_data = _extract_data(filepath)

    if not raw_data.strip():
        raise ValueError(f"File is empty: {filepath}")

    try:
        json.loads(raw_data, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid JSON in {filepath}: {exc}"
        ) from exc

    adapter: TypeAdapter[List[PromptDef]] = TypeAdapter(
        Annotated[List[PromptDef], Field(min_length=1)]
    )
    try:
        result: List[PromptDef] = adapter.validate_json(raw_data)
        return result
    except ValidationError as exc:
        raise ValueError(
            f"Schema validation error in {filepath}: {exc}"
        ) from exc
