"""Execution pipeline for constrained function calling."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .constrained_decoder import ConstrainedDecoder
from .models import FunctionCallResult
from .parser import parse_function_definitions, parse_prompts


def main() -> None:
    """Run constrained decoding over test prompts and output JSON."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to write output JSON file.",
    )
    args = parser.parse_args()

    # --- Load input files ---
    try:
        functions_def = parse_function_definitions(args.functions_definition)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"Error loading function definitions: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        test_prompts = parse_prompts(args.input)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"Error loading test prompts: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Initialise model ---
    try:
        model = Small_LLM_Model()
    except Exception as exc:
        print(f"Error initialising LLM model: {exc}", file=sys.stderr)
        sys.exit(1)

    decoder = ConstrainedDecoder(model, functions_def)

    # --- Process each prompt ---
    results: List[Dict[str, Any]] = []
    for item in test_prompts:
        try:
            raw_call = decoder.build_dict(item.prompt)
            validated = FunctionCallResult.model_validate(raw_call)
            results.append(validated.model_dump())
        except Exception as exc:
            print(
                f"Warning: failed prompt '{item.prompt}': {exc}",
                file=sys.stderr,
            )
            results.append({
                "prompt": item.prompt,
                "name": "",
                "parameters": {},
            })

    # --- Write output ---
    try:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")
    except (OSError, PermissionError) as exc:
        print(f"Error writing output file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
