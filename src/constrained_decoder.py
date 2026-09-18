"""Constrained decoder: forces 100% valid JSON via logit masking."""

import json
import textwrap
from typing import Dict, List, Tuple, Any

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .models import FunctionDef


class ConstrainedDecoder:
    """Generate structured function-call JSON using constrained decoding.

    At each token step, this class masks out all tokens that would
    break the JSON schema, then picks the highest-scoring valid token.
    This guarantees 100% parseable output regardless of model quality.
    """

    def __init__(self, model: Small_LLM_Model,
                 functions_def: List[FunctionDef]) -> None:
        """Initialise decoder with an LLM and function definitions.

        Args:
            model: An instance of Small_LLM_Model.
            functions_def: List of validated FunctionDef objects.
        """
        self.model = model
        self.functions_def = functions_def
        self.fct_names = [f.name for f in self.functions_def]
        self.vocab_path = self.model.get_path_to_vocab_file()

        self.fct_catalog = '\n'.join([
            f"- {f.name}: {f.description} "
            f"(params: {', '.join(f'{k}: {v.type}' for k, v in f.parameters.items())})"  # noqa: E501
            for f in self.functions_def
        ])

        with open(self.vocab_path, "r", encoding='utf-8') as f:
            raw_vocab: Dict[str, int] = json.load(f)

        # Invert and normalize space/newline markers
        self.vocab: Dict[int, str] = {
            int(v): k.replace("\u0120", " ").replace("\u010a", "\n")
            for k, v in raw_vocab.items()
        }

        # Pre-filter numeric token IDs using normalized strings
        valid_num_chars = set("0123456789.-")
        self.number_token_ids = [
            tid for tid, s in self.vocab.items()
            if s.strip() and all(c in valid_num_chars for c in s.strip())
        ]

        valid_int_chars = set("0123456789-")
        self.integer_token_ids = [
            tid for tid, s in self.vocab.items()
            if s.strip() and all(c in valid_int_chars for c in s.strip())
        ]

        self.stop_token_ids = [
            tid for tid, s in self.vocab.items()
            if s.strip() in (",", "}")
        ]

        # Prefix cache to avoid timeouts during name decoding
        self._prefix_cache: Dict[
            Tuple[str, Tuple[str, ...]], List[int]
        ] = {}

    def _build_prompt(self, user_prompt: str) -> str:
        """Build the LLM prompt with function catalog and JSON template.

        Args:
            user_prompt: The user's natural language request.

        Returns:
            A formatted prompt string ready for tokenization.
        """
        final_prompt = f"""\
            You are a function-calling assistant.
            Output JSON with the correct function name and arguments.

            Available functions:
            {self.fct_catalog}

            User prompt: {user_prompt}

            JSON:
            {{
                "prompt": {json.dumps(user_prompt)},
                "name": \\\""""
        return textwrap.dedent(final_prompt)

    def _encode_tolist(self, text: str) -> List[int]:
        """Tokenize text into a flat list of token IDs.

        Args:
            text: The string to encode.

        Returns:
            A list of integer token IDs.
        """
        tensor_ids = self.model.encode(text)
        return [int(x) for x in tensor_ids[0].tolist()]

    def _get_valid_prefix_ids(
        self, built: str, targets: List[str]
    ) -> List[int]:
        """Find token IDs that continue a valid prefix toward any target.

        Args:
            built: The string built so far.
            targets: List of allowed complete strings.

        Returns:
            List of token IDs whose text keeps built as a valid prefix.
        """
        cache_key = (built, tuple(targets))
        if cache_key in self._prefix_cache:
            return self._prefix_cache[cache_key]

        valid: List[int] = []
        for tid, token_str in self.vocab.items():
            candidate = built + token_str
            for target in targets:
                if target.startswith(candidate):
                    valid.append(tid)
                    break
        self._prefix_cache[cache_key] = valid
        return valid

    def function_name_finding(self, input_ids: List[int]) -> str:
        """Constrained-decode a function name using prefix matching.

        At each step only tokens that keep the generated string as a
        valid prefix of a known function name are allowed.

        Args:
            input_ids: Current token ID context (modified in-place).

        Returns:
            The predicted function name string.
        """
        predicted_name = ""
        max_len = max((len(f) for f in self.fct_names), default=20)

        for _ in range(max_len + 5):
            valid_ids = self._get_valid_prefix_ids(
                predicted_name, self.fct_names
            )
            if not valid_ids:
                break

            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids)
            ).flatten()
            mask = np.full(logits.shape, -np.inf)

            for vid in valid_ids:
                if vid < len(mask):
                    mask[vid] = logits[vid]

            # If current name already matches a function but could
            # also extend to a longer one (e.g. "fn_add" vs
            # "fn_add_numbers"), unmask stop tokens so the model
            # can choose to stop here instead of being forced to
            # continue toward the longer name.
            if predicted_name in self.fct_names:
                for sid in self.stop_token_ids:
                    if sid < len(mask):
                        mask[sid] = logits[sid]

            chosen_id = int(np.argmax(mask))
            chosen_str = self.vocab.get(chosen_id, "")

            # Model chose a stop token: keep the current name
            if chosen_str.strip() in (",", "}", "\n", ""):
                break

            predicted_name += chosen_str
            input_ids.append(chosen_id)

            if predicted_name in self.fct_names:
                # Only auto-stop if no longer name is possible
                has_longer = any(
                    n != predicted_name
                    and n.startswith(predicted_name)
                    for n in self.fct_names
                )
                if not has_longer:
                    break
                    
        # Fallback: if decoding produced no valid name, pick first
        if predicted_name not in self.fct_names:
            predicted_name = self.fct_names[0]

        return predicted_name

    def _generate_numbers(self, input_ids: List[int]) -> str:
        """Constrained-decode a numeric value (float).

        Enforces: only digits, at most one dot, minus only at start.
        Stops on comma, brace, or newline.

        Args:
            input_ids: Current token ID context (modified in-place).

        Returns:
            String representation of the decoded number.
        """
        predicted_val = ""
        for _ in range(15):
            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids)
            ).flatten()
            mask = np.full(logits.shape, -np.inf)

            for tid in self.number_token_ids:
                tok = self.vocab[tid].strip()
                ok = True
                for ch in tok:
                    if ch == '.' and '.' in predicted_val:
                        ok = False
                        break
                    if ch == '-' and predicted_val:
                        ok = False
                        break
                if ok and tid < len(mask):
                    mask[tid] = logits[tid]

            # Only allow stop tokens if we have at least one digit
            if predicted_val and any(c.isdigit() for c in predicted_val):
                for sid in self.stop_token_ids:
                    if sid < len(mask):
                        mask[sid] = logits[sid]

            chosen_id = int(np.argmax(mask))
            token_str = self.vocab.get(chosen_id, "")
            stripped = token_str.strip()

            if stripped in (',', '}', ''):
                break

            predicted_val += stripped
            input_ids.append(chosen_id)

        # BUG FIX: validate the final string can be parsed as float
        try:
            float(predicted_val)
        except (ValueError, TypeError):
            predicted_val = "0"

        return predicted_val

    def _generate_integers(self, input_ids: List[int]) -> str:
        """Constrained-decode an integer value.

        Enforces: only digits, minus only at start.
        Stops on comma, brace, or newline.

        Args:
            input_ids: Current token ID context (modified in-place).

        Returns:
            String representation of the decoded integer.
        """
        predicted_val = ""
        for _ in range(15):
            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids)
            ).flatten()
            mask = np.full(logits.shape, -np.inf)

            for tid in self.integer_token_ids:
                tok = self.vocab[tid].strip()
                ok = True
                for ch in tok:
                    if ch == '-' and predicted_val:
                        ok = False
                        break
                if ok and tid < len(mask):
                    mask[tid] = logits[tid]

            if predicted_val and any(c.isdigit() for c in predicted_val):
                for sid in self.stop_token_ids:
                    if sid < len(mask):
                        mask[sid] = logits[sid]

            chosen_id = int(np.argmax(mask))
            token_str = self.vocab.get(chosen_id, "")
            stripped = token_str.strip()

            if stripped in (',', '}', ''):
                break

            predicted_val += stripped
            input_ids.append(chosen_id)

        # BUG FIX: validate the final string can be parsed as int
        try:
            int(predicted_val)
        except (ValueError, TypeError):
            predicted_val = "0"

        return predicted_val

    def _generate_boolean(self, input_ids: List[int]) -> bool:
        """Constrained-decode a boolean value (true/false).

        Uses prefix matching against the literals 'true' and 'false'.

        Args:
            input_ids: Current token ID context (modified in-place).

        Returns:
            The decoded boolean value.
        """
        allowed = ["true", "false"]
        built = ""
        for _ in range(6):
            valid_ids = [
                tid for tid, s in self.vocab.items()
                if any(
                    a.startswith((built + s).strip())
                    for a in allowed
                )
            ]
            if not valid_ids:
                break

            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids)
            ).flatten()
            mask = np.full(logits.shape, -np.inf)
            for vid in valid_ids:
                if vid < len(mask):
                    mask[vid] = logits[vid]

            chosen_id = int(np.argmax(mask))
            chosen_str = self.vocab.get(chosen_id, "")
            built += chosen_str
            input_ids.append(chosen_id)

            if built.strip() in allowed:
                break

        return built.strip() == "true"

    def _generate_string(
        self, input_ids: List[int], max_steps: int = 50
    ) -> str:
        """Constrained-decode a string value (stops at closing quote).

        Masks out newline tokens. When a token containing a double-quote
        is chosen, only the part before the quote is kept.

        Args:
            input_ids: Current token ID context (modified in-place).
            max_steps: Maximum number of tokens to generate.

        Returns:
            The decoded string content (without surrounding quotes).
        """
        built = ""
        for _ in range(max_steps):
            logits = np.array(
                self.model.get_logits_from_input_ids(input_ids)
            ).flatten()

            # Mask out newlines and unescaped quotes
            mask = np.full(logits.shape, -np.inf)
            for tid, s in self.vocab.items():
                if "\n" not in s and tid < len(mask):
                    mask[tid] = logits[tid]

            chosen_id = int(np.argmax(mask))
            token_str = self.vocab.get(chosen_id, "")

            if '"' in token_str:
                idx = token_str.index('"')
                built += token_str[:idx]
                break

            built += token_str
            input_ids.append(chosen_id)

        return built.strip()

    def build_dict(self, user_prompt: str) -> Dict[str, Any]:
        """Produce a function-call dict for the given user prompt.

        This is the main entry point. It builds a prompt, constrained-
        decodes the function name, then constrained-decodes each
        parameter value according to its type from the schema.

        Args:
            user_prompt: The user's natural language request.

        Returns:
            A dict with keys: prompt, name, parameters.
        """
        prompt_text = self._build_prompt(user_prompt)
        input_ids = self._encode_tolist(prompt_text)

        # 1. Decode function name
        fct_name = self.function_name_finding(input_ids)

        # 2. Append JSON structure bridge
        bridge = '",\n"parameters": {'
        input_ids.extend(self._encode_tolist(bridge))

        # 3. Find matching function definition
        target_fn = next(
            (f for f in self.functions_def if f.name == fct_name),
            None,
        )
        params_result: Dict[str, Any] = {}

        if target_fn and target_fn.parameters:
            items = list(target_fn.parameters.items())
            for i, (p_name, p_def) in enumerate(items):
                key_prefix = f'"{p_name}": '
                input_ids.extend(self._encode_tolist(key_prefix))

                val_serialized = '""'

                if p_def.type == 'number':
                    val_str = self._generate_numbers(input_ids)
                    params_result[p_name] = float(val_str)
                    val_serialized = val_str

                elif p_def.type == 'integer':
                    val_str = self._generate_integers(input_ids)
                    params_result[p_name] = int(val_str)
                    val_serialized = val_str

                elif p_def.type == 'string':
                    input_ids.extend(self._encode_tolist('"'))
                    val_str = self._generate_string(input_ids)
                    params_result[p_name] = val_str
                    val_serialized = f'"{val_str}"'
                    input_ids.extend(self._encode_tolist('"'))

                elif p_def.type == 'boolean':
                    bool_val = self._generate_boolean(input_ids)
                    params_result[p_name] = bool_val
                    val_serialized = "true" if bool_val else "false"

                else:
                    # Unknown type: default to empty string
                    params_result[p_name] = ""

                # BUG FIX: feed the generated value back into context
                # so the model sees previous param values when
                # generating the next one (string already does this
                # inside _generate_string, so we skip it here)
                if p_def.type in ('number', 'integer', 'boolean'):
                    input_ids.extend(
                        self._encode_tolist(val_serialized)
                    )

                if i + 1 < len(items):
                    input_ids.extend(self._encode_tolist(', '))

        return {
            "prompt": user_prompt,
            "name": fct_name,
            "parameters": params_result
        }
