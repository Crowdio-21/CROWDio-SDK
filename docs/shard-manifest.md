# Shard manifest v1

Shared by the SDK (export tool), the Foreman (shard catalogue) and workers.
Machine-readable schema: [`manifest.schema.json`](manifest.schema.json).
Reference implementation and the structural rules JSON Schema cannot express:
`crowdio.catalogue.validate_manifest`.

## Layout

```
<model_id>/
  manifest.json              written only after verification passes
  storage_report.json
  verification_report.json
  shards/<shard_id>.onnx     (+ <shard_id>.onnx.data when external data is used)
  tokenizer/
```

## Atoms and shards

- Transformer layers are grouped into atoms of `cut_grid` layers; the last atom
  is shorter when the layer count is not a multiple. `atom_map` lists each
  atom's inclusive layer range.
- `encoder` shards cover every contiguous atom range: A(A+1)/2 when
  `complete_encoder_ranges` is true, otherwise a subset that still covers every atom.
- `embed` and `head` are `pinned`, carry no atom or layer bounds, and are never
  merged with encoder ranges. Exactly one of each per precision.
- `full` is the whole model (embed, every layer, head) and is exported when it
  fits `max_shard_bytes`.
- Shard ids: `<model_id>.embed`, `<model_id>.enc.a<s>-a<e>`, `<model_id>.head`,
  `<model_id>.full`; non-fp32 variants append `@<precision>`.

## Tensor contract

| role    | inputs                                   | outputs                                      |
|---------|------------------------------------------|----------------------------------------------|
| embed   | `input_ids` int64 [batch, seq], `attention_mask` int64 [batch, seq] | `hidden_states_out` f32 [batch, seq, d], `attention_mask_out` int64 |
| encoder | `hidden_states` f32 [batch, seq, d], `attention_mask` int64 | `hidden_states_out`, `attention_mask_out` |
| head    | `hidden_states`                          | `logits` f32 [batch, num_labels]             |
| full    | `input_ids`, `attention_mask`            | `logits`                                     |

Binding rule: an output's `binds_to` names the next shard's input. A runtime
feeds a shard only the inputs it declares (so the head ignores the mask).
`attention_mask` is 1 for real tokens and 0 for padding; it is carried through
every encoder shard unchanged. BERT shards assume single-segment input
(`token_type_ids` = 0).

## Sizes and cost fields

- `bytes`: size of the main `.onnx` file; `external_data_files` lists any
  `.data` files with their own size and sha256.
- `flops`: `flops_estimation` states the method (`analytic_matmul_v1`, FLOPs per
  sequence at `reference_seq_len`).
- `in_act_bytes_per_token` / `out_act_bytes_per_token`: fp32 hidden states plus
  int64 ids/mask per token. Extension: `out_act_bytes_per_sequence` for the
  per-sequence logits of `head` and `full`.

## Changes to the shared contract (for the other agents)

- Added fields: `file`, `out_act_bytes_per_sequence`, tensor `binds_to`,
  manifest `version`, `complete_encoder_ranges`, `flops_estimation`.
- `external_data_files` entries are objects `{path, bytes, sha256}`, not bare names.
- Embed and head carry `null` atom and layer bounds.
