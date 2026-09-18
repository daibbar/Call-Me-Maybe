"""Pydantic models for function definitions and call results."""

from pydantic import BaseModel, ConfigDict
from typing import Any, Dict


class ParametersDef(BaseModel):
    """Definition of a single function parameter.

    Attributes:
        type: The expected type (number, string, boolean, integer).
    """

    model_config = ConfigDict(extra="forbid")
    type: str


class ReturnsDef(BaseModel):
    """Definition of a function return type.

    Attributes:
        type: The return type string.
    """

    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionDef(BaseModel):
    """Complete function definition from the input schema.

    Attributes:
        name: The function name (e.g. fn_add_numbers).
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter name to its ParametersDef.
        returns: The return type definition.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ParametersDef]
    returns: ReturnsDef


class PromptDef(BaseModel):
    """A single test prompt from the input file.

    Attributes:
        prompt: The natural-language user request.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallResult(BaseModel):
    """Output schema for a single function call result.

    Attributes:
        prompt: The original user prompt.
        name: The selected function name.
        parameters: The extracted arguments with correct types.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: Dict[str, Any]
