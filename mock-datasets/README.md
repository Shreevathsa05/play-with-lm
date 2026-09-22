# Mock datasets for objective readiness

Upload these JSON files on **Data preparation**, wait for audit `COMPLETED`, then open the report. Each file targets specific **READINESS** objectives.

## SFT (already covered)

| File | Input shape | Unlocks |
|------|-------------|---------|
| `fullparams_sft.json` | OpenAI `messages` | **sft** |
| `lora_sft.json` | Alpaca `instruction` / `input` / `output` | **sft** |
| `qlora_sft.json` | ShareGPT `conversations` | **sft** |

## Other objectives

| File | Schema after audit | Unlocks |
|------|--------------------|---------|
| `pretrain_cpt_raw_text.json` | `raw_text` (`text`) | **pretrain**, **cpt** |
| `dpo_reward_paired.json` | `paired_preference` (`prompt`, `chosen`, `rejected`) | **dpo**, **reward_model** |
| `kto_binary.json` | `binary_preference` (`prompt`, `completion`, `label` bool) | **kto** |
| `ppo_grpo_prompts.json` | `prompt_only` (`prompt`) | **ppo**, **grpo** |
| `embedding_triplets.json` | `embedding_triplet` (`anchor`, `positive`, `negative`) | **embedding** |
| `embedding_pairs.json` | `embedding_pair` (`text1`, `text2`) | **embedding** |

## How to validate

1. Upload one file → wait until status is `COMPLETED`.
2. Open the audit report → **READINESS** should show `READY` for the objectives in the table (others stay `BLOCKED`).
3. On Fine-tune, pick a matching objective + a small model (e.g. `google/gemma-3-270m-it` or an embedding model for embedding jobs).

**Notes**

- PPO still needs a `rewardModelId` when you create the job.
- One dataset unlocks only the objectives that match its schema; that is expected.
- Keep files small; these are smoke fixtures, not production corpora.
