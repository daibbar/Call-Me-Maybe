from pydantic import BaseModel, ConfigDict
from typing import Dict, Union


class ParametersDef(BaseModel):
    type: str


class ReturnsDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, ParametersDef]
    returns: ReturnsDef


class PromptDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Union[float, str, bool, int, None]]
