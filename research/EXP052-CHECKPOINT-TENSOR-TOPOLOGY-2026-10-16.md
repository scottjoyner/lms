# EXP-052 addendum — Exact stored checkpoint tensor counts

**2026-10-16 | Read-only Safetensors shape metadata, not forward inference.**
For checkpoint custody, see frozen EXP051 and EXP052 JSON receipts and audit scripts.
Values below are exact `sum(prod(tensor.shape))` over unique serialized tensor keys, obtained with `safetensors.safe_open(...).get_slice(...).get_shape()`, rather than loading model tensors into RAM.

| Checkpoint | Stored scalars | BF16 scalars | F32 scalars | Serialized tensors |
|---|---:|---:|---:|---:|
| K2-Horizon-3.7B | 5,058,255,360 | 5,058,255,360 | 0 | 327 |
| Qwen3.5-4B-Base | 4,659,865,088 | 4,659,861,248 | 3,840 | 738 |
| MiniCPM-V 4.7-1B | 1,554,476,272 | 1,554,476,272 | 0 | 780 |

## Exact top-level partitions

- **K2:** `model.layers` 3,775,057,920; `model.embed_tokens` 641,597,440; `lm_head.weight` 641,597,440; `model.norm` 2,560.
- **Qwen Base:** `model.language_model` 4,205,751,296; `model.visual` 333,514,240; `mtp.*` 120,599,552. The language group includes its tied token embedding, unlike K2's separate head.
- **MiniCPM:** `model.language_model` 752,161,600; `lm_head.weight` 254,048,256; `model.vision_tower` 522,299,312; `model.merger` 25,967,104.

## Interpretation and limits

- The K2 **3.7B** label is approximately consistent with the 3.775B serialized decoder-layer body; its complete checkpoint is 5.058B stored scalars because embedding and LM head are each 641.6M.
- The Qwen **4B** text block is 4.206B including its tied embedding, while 333.5M vision and 120.6M MTP scalars are also serialized. MiniCPM **1B** is approximately 1.006B text+output head, with 548.3M vision/merger.
- Saved scalars are not guaranteed independent trainable or active-at-token parameters; tensor aliases, tied weights, MTP heads and vision-only components must be distinguished before claiming equal parameter budgets.
- These three EXP051/052 **text-only** studies provide no measurement of vision/OCR/video performance, MoE expert activation, full application efficiency or K2↔Qwen neuron homology.
- No checkpoint data saved in Git; no model modifications, production changes, routing, training, deployment or arXiv.