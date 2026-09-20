ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS objective VARCHAR(255) NOT NULL DEFAULT 'sft';
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS reward_model_id VARCHAR(255);
ALTER TABLE finetune_jobs ADD COLUMN IF NOT EXISTS resume_from_checkpoint TEXT;
