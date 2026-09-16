*This project has been created as part of the 42 curriculum by mdaibbar.*



## Ressources
uv: https://www.youtube.com/watch?v=AMdG7IjgSPM&t=599s




```python
                                [functions_definition.json]
                                              │
                                              ▼
[Prompt String] ──> model.encode() ──> [Input IDs] ──> model.get_logits_from_input_ids()
                                                               │
                                                               ▼
                                                        [Raw Logits Vector]
                                                               │
      [vocab.json] ──> ID-to-String Map ──> Schema Filter ─────┼──> Mask invalid to -inf
                                                               │
                                                               ▼
                                                      [Filtered Next Token]
                                                               │
                                                               ▼
                                                       (Append & Repeat)
                                                               │
                                                               ▼
                                            [function_calling_results.json]
```


"Our architecture uses template-filling constrained decoding via prefix forcing, where the invariant JSON schema is injected deterministically into context, and the variable slots (function names and typed parameters) are resolved via string-prefix logit masking."