from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Dict, Literal, Annotated

valid_str = Annotated[
    str,
    StringConstraints(min_length=1, strip_whitespace=True)
]

class ParametersDef(BaseModel):

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "integer", "boolean"]

class ReturnsDef(BaseModel):

    model_config = ConfigDict(extra="forbid")
    type: valid_str

class FunctionDef(BaseModel):

    model_config = ConfigDict(extra="forbid")
    name: valid_str
    description: valid_str
    parameters: Dict[valid_str, ParametersDef]
    returns: ReturnsDef

class PromptDef(BaseModel):

    model_config = ConfigDict(extra="forbid")
    prompt: valid_str
