from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionDef
from typing import List, Dict
import json
import textwrap
import numpy as np


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model, 
                 functions_def: List[FunctionDef]) -> None:
        self.model = model
        self.functions_def = functions_def
        self.vocab_path = self.model.get_path_to_vocab_file()
        self.fct_catalog = '\n'.join([
            f"-{f.name}: {f.description} "
            f"(params: {','.join(f'{k}: {v.type}' for k, v in f.parameters.items())})"
            for f in self.functions_def
        ])

        # init the vocab: open the vocab file, invert and replace
        with open(self.vocab_path, "r", encoding='utf-8') as f:
            raw_vocab: Dict[str, int] = json.load(f)

        self.vocab = {
            int(v): k.replace("Ġ", " ").replace("Ċ", "\n") 
            for k, v in raw_vocab.items()
        }

        valid_chars = set("0123456789-.")
        self.number_token_ids = [
            v for k, v in raw_vocab.items()
            if k and all(c in valid_chars for c in k)
        ]

        self.stop_token_ids = [
            tid for tid, s in self.vocab.items()
            if s.strip() in (",", "}")
        ]

        valid_int_chars = set("0123456789-")
        self.integer_token_ids = [
            v for k, v in raw_vocab.items()
            if k and all(c in valid_int_chars for c in k)
        ]

    def _build_prompt(self, user_prompt: str) -> str:
        final_prompt = f"""\
            You are a function-calling assistant that
            helps me get a JSON format from a user prompt.

            Available functions:
            {self.fct_catalog}

            User prompt: {user_prompt}

            JSON:
            {{
                "prompt": {json.dumps(user_prompt)},
                "name": \""""

        return textwrap.dedent(final_prompt)

    def function_name_finding(self, prompt: str) -> str:
        tensor_ids = self.model.encode(prompt)
        input_ids = [int(x) for x in tensor_ids[0].tolist()]
        raw_logits = self.model.get_logits_from_input_ids(input_ids)
        logits = np.array(raw_logits).flatten()

        mask = np.full(logits.shape, -np.inf)
        fct_names = [fct_def.name for fct_def in self.functions_def]
        fct_name_predection = ""

        max_fct_name_len = max(len(f) for f in fct_names)
        for _ in range(max_fct_name_len):
            for token_id, token_str in self.vocab.items():
                candidate = fct_name_predection + token_str
                for fct_name in fct_names:
                    if fct_name.startswith(candidate) or fct_name == candidate:
                        if token_id < len(mask):
                            mask[token_id] = logits[token_id]
                        break

            chosen_token_id = int(np.argmax(mask))
            chosen_token_str = self.vocab.get(chosen_token_id, "")

            fct_name_predection += chosen_token_str
            input_ids.append(chosen_token_id)

            if fct_name_predection in fct_names:
                break

            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits).flatten()
            mask = np.full(logits.shape, -np.inf)

        return fct_name_predection

    def _add_params_template(self, prompt: str) -> str:
        return prompt.strip() + '",\n"parameters": {'

    def _encode_tolist(self, prompt: str) -> List[int]:
        tensor_ids = self.model.encode(prompt)
        input_ids = [int(x) for x in tensor_ids[0].tolist()]
        return input_ids

    def _generate_numbers(self, input_ids: List[int]) -> str:
        predicted_val = ""

        for _ in range(15):
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits).flatten()
            mask = np.full(logits.shape, -np.inf)

            for candidate in self.number_token_ids:
                if '.' in self.vocab[candidate] and '.' in predicted_val:
                    continue
                mask[candidate] = logits[candidate]

            if predicted_val:
                for candidate in self.stop_token_ids:
                    mask[candidate] = logits[candidate]

            chosen_id = int(np.argmax(mask))
            prediction = self.vocab[chosen_id].strip()
            if prediction in (',', '}'):
                break
            predicted_val += prediction
            input_ids.append(chosen_id)
        return predicted_val or '0'

    def _generate_integers(self, input_ids: List[int]) -> str:
        predicted_val = ""

        for _ in range(15):
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits).flatten()
            mask = np.full(logits.shape, -np.inf)

            for candidate in self.integer_token_ids:
                mask[candidate] = logits[candidate]

            if predicted_val:
                for candidate in self.stop_token_ids:
                    mask[candidate] = logits[candidate]

            chosen_id = int(np.argmax(mask))
            prediction = self.vocab[chosen_id].strip()
            if prediction in (',', '}'):
                break
            predicted_val += prediction
            input_ids.append(chosen_id)
        return predicted_val or '0'

    def _generate_boolean(self, input_ids: List[int], max_steps: int = 10) -> bool:
        allowed = ["true", "false"]
        built = ""

        for _ in range(max_steps):
            valid_ids = [
                tid for tid, s in self.vocab.items()
                if any(a.startswith(built + s.strip()) for a in allowed)
            ]
            if not valid_ids:
                break

            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits).flatten()
            mask = np.full(logits.shape, -np.inf)
            for vid in valid_ids:
                mask[vid] = logits[vid]

            chosen_id = int(np.argmax(mask))
            token_str = self.vocab.get(chosen_id, "").strip()
            built += token_str
            input_ids.append(chosen_id)
            if built in allowed:
                break

        return built == "true"

    def _generate_string(self, input_ids: List[int], max_steps: int = 30) -> str:
        valid_ids = [
            tid for tid, s in self.vocab.items()
            if '"' not in s and "\n" not in s
        ]

        built = ""

        for _ in range(max_steps):
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(raw_logits).flatten()

            top_id = int(np.argmax(logits))
            top_str = self.vocab.get(top_id, "")
            if '"' in top_str:
                idx = top_str.index('"')
                built += top_str[:idx]
                break

            mask = np.full(logits.shape, -np.inf)
            for vid in valid_ids:
                mask[vid] = logits[vid]

            chosen_id = int(np.argmax(mask))
            token_str = self.vocab.get(chosen_id, "")
            built += token_str
            input_ids.append(chosen_id)

        return built.strip()

    def build_dict(self, user_prompt: str) -> dict:
        input_prompt = self._build_prompt(user_prompt)
        fct_name_predection = self.function_name_finding(input_prompt)
        final_prompt = self._add_params_template(input_prompt + fct_name_predection)
        params_result = {}

        for f in self.functions_def:
            if f.name == fct_name_predection:
                if len(f.parameters) == 0:
                    break

                final_prompt_ids = self._encode_tolist(final_prompt)
                for i, p in enumerate(f.parameters):
                    param = f'"{p}": '
                    final_prompt_ids.extend(self._encode_tolist(param))

                    if f.parameters[p].type == 'number':
                        val_prediction = self._generate_numbers(final_prompt_ids)
                        params_result[p] = float(val_prediction)
                    elif f.parameters[p].type == 'integer':
                        val_prediction = self._generate_integers(final_prompt_ids)
                        params_result[p] = int(val_prediction)
                    elif f.parameters[p].type == 'string':
                        val_prediction = self._generate_string(final_prompt_ids)
                        params_result[p] = val_prediction
                    elif f.parameters[p].type == 'boolean':
                        bool_val = self._generate_boolean(final_prompt_ids)
                        params_result[p] = bool_val
                        val_prediction = "true" if bool_val else "false"
                    else:
                        val_prediction = '""'

                    if i + 1 < len(f.parameters):
                        final_prompt_ids.extend(self._encode_tolist(str(val_prediction) + ', '))

        result = {
            "prompt": user_prompt,
            "name": fct_name_predection,
            "parameters": params_result
        }

        return result