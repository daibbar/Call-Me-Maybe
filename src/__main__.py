from llm_sdk import Small_LLM_Model
from typing import List
from models import FunctionDef
from parser import parse_function_definitions, parse_prompts
from constrained_decoder import ConstrianedDecoder


try:
    functions_def = parse_function_definitions(functions_path)
    prompts = parse_prompts(prompts_path)

except (ValueError, PermissionError, FileNotFoundError) as e:
    pass

constrained_decoder = ConstrianedDecoder(Small_LLM_Model, functions_def, prompts)







