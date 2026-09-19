import argparse
import json
import os
import sys
from typing import Any, Dict, List

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .constrained_decoder import ConstrainedDecoder
from .parser import parse_function_definitions, parse_prompts

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    # --- Load input files ---
    try:
        functions_def = parse_function_definitions(args.functions_definition)
    except (OSError, ValueError) as exc:
        print(f"Error loading function definitions: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        test_prompts = parse_prompts(args.input)
    except (OSError, ValueError) as exc:
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
            call_dict = decoder.build_dict(item.prompt)
            results.append(call_dict)
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
    except OSError as exc:
        print(f"Error writing output file: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"an error has occured: {e}", file=sys.stderr)
    except KeyboardInterrupt as e:
        print(f"you quit the program, GoodBye!! {e}")
