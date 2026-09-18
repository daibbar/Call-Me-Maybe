*This project has been created as part of the 42 curriculum by mdaibbar.*

# Call Me Maybe

## Description

A function-calling system that translates natural language prompts into structured JSON function calls using constrained decoding with a small LLM (Qwen3-0.6B, 600M parameters).

Given a prompt like *"What is the sum of 2 and 3?"*, the system outputs:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The key insight is that we **do not rely on the model to spontaneously produce valid JSON**. Instead, we use **constrained decoding** — at every token generation step, we mask out all tokens that would break the JSON structure or violate the function schema, then let the model choose only from valid tokens.

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
make install
# or: uv sync
```

### Running

```bash
# Default paths (reads from data/input/, writes to data/output/)
make run

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Commands

```bash
make debug   # Run with Python debugger (pdb)
make lint    # Run flake8 + mypy
make clean   # Remove __pycache__, .mypy_cache, output
```

## Algorithm Explanation

### How Constrained Decoding Works

Language models generate text one token at a time. At each step the model produces a **logits** vector — a score for every possible next token in its vocabulary (~150k tokens for Qwen3).

Normally, you pick the token with the highest logit. But this gives no guarantee the output will be valid JSON or match a schema.

**Constrained decoding** fixes this by intervening before token selection:

1. Get the logits from the model for the current context
2. Check which tokens would keep the output valid (correct JSON structure + matching the function schema)
3. Set all invalid token logits to `-infinity`
4. Pick the highest-scoring token from the remaining valid ones
5. Append it to the context and repeat

### Our Pipeline Step by Step

```
[Prompt String] --> model.encode() --> [Input IDs] --> model.get_logits_from_input_ids()
                                                            |
                                                            v
                                                     [Raw Logits Vector]
                                                            |
      [vocab.json] --> ID-to-String Map --> Schema Filter ---+--> Mask invalid to -inf
                                                            |
                                                            v
                                                   [Filtered Next Token]
                                                            |
                                                            v
                                                    (Append & Repeat)
                                                            |
                                                            v
                                         [function_calling_results.json]
```

**Phase 1 — Function Name Decoding:** We use prefix matching. At each step, we only allow tokens that keep the built string as a valid prefix of at least one known function name. This guarantees the output is always an exact function name.

**Phase 2 — Parameter Decoding:** For each parameter in the matched function's schema:
- **number**: Only allow tokens containing digits, dots, and minus signs. Enforce at most one dot, minus only at position 0. Stop on comma/brace tokens.
- **integer**: Same as number but no dot allowed.
- **string**: Allow all tokens except newlines. Stop when a double-quote token appears (closing the JSON string).
- **boolean**: Prefix-match against the literals `true` and `false`.

Between each step, the JSON structure (`"key":`, commas, braces) is injected deterministically — the model never generates these structural tokens.

## Design Decisions

1. **Template-filling approach**: Rather than having the model generate the entire JSON, we inject the fixed structure (braces, keys, colons) ourselves and only let the model generate the variable parts (function name, parameter values). This makes the output 100% valid JSON by construction.

2. **Greedy decoding (argmax)**: We always pick the highest-scoring valid token. No sampling, no temperature. This is deterministic and simple to explain.

3. **Pydantic for all validation**: Input files, function definitions, prompts, and output results all use pydantic models with `extra="forbid"` to catch schema violations early.

4. **Vocabulary pre-filtering**: At init time, we pre-compute which token IDs contain only numeric characters. This avoids scanning the entire 150k vocabulary at every number-generation step.

5. **Prefix caching**: The `_get_valid_prefix_ids` results are cached by `(built_string, targets)` to avoid redundant vocabulary scans during function name decoding.

## Performance Analysis

- **JSON validity**: 100% — guaranteed by construction, not by prompting
- **Function name accuracy**: Very high — prefix matching eliminates hallucinated names
- **Argument extraction**: Depends on model quality; the 0.6B model handles simple numeric and string extraction well
- **Speed**: Each prompt requires multiple forward passes (one per generated token); total runtime for 11 test prompts is under 5 minutes on CPU

## Challenges Faced

1. **Token representation**: The vocabulary file uses special markers (`Ġ` for space, `Ċ` for newline) that must be normalized before string comparison.

2. **Multi-character tokens**: A single token might contain multiple characters (e.g., `"123"` or `" hello"`). Number validation must check each character in the token, not just the first one.

3. **Context feeding**: After generating a parameter value, the tokens must be fed back into the input context so the model "sees" previous values when generating the next parameter. Without this, multi-parameter functions get poor accuracy.

4. **Edge cases in number parsing**: Tokens like `"-"` or `"."` alone are valid number characters but not valid numbers. Added try/except fallbacks to handle these.

## Testing Strategy

1. **Input validation**: Tested with missing files, empty files, invalid JSON, and wrong-schema JSON to verify graceful error messages.

2. **Output validation**: Verified that every output is valid JSON parseable by `json.load()`, with correct keys and types matching the function definitions.

3. **Edge cases tested**: Empty strings, large numbers, negative numbers, multi-parameter functions, ambiguous prompts.

4. **Lint compliance**: Ran `make lint` (flake8 + mypy with strict flags) to verify code quality.

## Example Usage

```bash
# Run with default test files
$ uv run python -m src
Results written to data/output/function_calling_results.json

# Run with custom inputs
$ uv run python -m src --input my_prompts.json --output my_results.json

# Check the output
$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  ...
]
```

## Resources

- [Constrained Decoding for LLMs (HuggingFace blog)](https://huggingface.co/blog/constrained-beam-search)
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [BPE Tokenization Explained](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [uv Package Manager](https://docs.astral.sh/uv/)

### AI Usage Disclosure

AI tools were used during development for:
- Exploring constrained decoding concepts and understanding the theory
- Debugging token mapping issues with the vocabulary file
- Drafting initial test cases

All generated code was reviewed, understood, and rewritten by hand. The core constrained decoding logic was designed and implemented manually.