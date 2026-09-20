# Mock SFT datasets

Upload these JSON files on `/student/data`, wait for audit `COMPLETED`, then start a fine-tune job.

| File | Schema | Best for |
|------|--------|----------|
| `fullparams_sft.json` | OpenAI `messages` | `fullparams` |
| `lora_sft.json` | Alpaca `instruction` / `input` / `output` | `lora` |
| `qlora_sft.json` | ShareGPT `conversations` | `qlora_4bit`, `qlora_8bit` |

All three are valid for any SFT recipe (`fullparams`, `lora`, `qlora_4bit`, `qlora_8bit`). Suggested smoke settings: small instruct model (e.g. `google/gemma-3-270m-it`), `maxSeqLength` 512, `batchSize` 1.
