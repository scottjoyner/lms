# EXP-053 weights-only model identity preflight (research freeze, 2026-10-16)

## Purpose
Before any EXP053 human-review approval or model inference, fingerprint the exact local BF16 checkpoint files for three pre-identified families: K2-Horizon-3.7B, Qwen3.5-4B-Base, MiniCPM-V-4.7-1B. This experiment provides source provenance, NOT model validation, ranking, generation, decoding, benchmark timing, or human adjudication.

## Checkpoints
- K2 local checkout at /home/scott/models/research/K2-Horizon-3.7B
- Qwen pretrained checkpoint at HuggingFace hub revision 1001bb4d826a52d1f399e183466143f4da7b741b
- MiniCPM-V 4.7-1B at revision 286817f1132b998913a129a221c89b4402b4ab80
The above paths are for the x1-370 research host only, not deployment targets.

## Source-of-truth file rules
- Require exactly the 3 pinned tokenizer/config SHA256 identities already published in EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json, whose report SHA256 is 8e35a44e1ebb725c0b52469baa4ab5d41472e38fe55346352b96dd26697726be.
- K2 and Qwen weight shard filenames are the set of weight_map values in their model.safetensors.index.json. Require each unique shard to exist and be nontrivial, and no unexpected *.safetensors files. Require at least 2 shard files. MiniCPM must have only model.safetensors and no index.
- Read each weight file as bytes in chunks, recording SHA256, size, filename, and the SHA256 of the index JSON where applicable. Refuse duplicate shards, missing files, path traversal, unexpected symlinks outside the model storage root, changing files, or changed tokenizer identity.
- Construct stable per-model digest from canonical sorted JSON consisting only of model label, revision, config SHA256, tokenizer SHA256, index SHA256, and per-file {name, bytes, sha256}. Record an overall 3-model manifest digest.
- Report the number of shards and their total bytes but not tensor values, tokenizer vocabulary, private prompts, RACE article text, gold answers, reviewer IDs, private keys, or individual model outputs.
- Dry run only validates fixed design. Actual scan is read-only. No neural model code is imported, no GPU/ROCm/CUDA touched, no weight reload/edits, no training, no routing, no NAS/Neo4j/production writes.
- Any missing or mismatched checkpoint means HARD HOLD on scientific promotion. Public SHA256 receipts are sufficient to check score archive checkpoint identity later; they do not prove the engine used the weights without independent runner attestation.
