CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY, email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL, role VARCHAR(255) NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id), filename VARCHAR(255) NOT NULL,
    minio_object_name VARCHAR(255), source VARCHAR(255), hugging_face_id VARCHAR(255), hugging_face_config VARCHAR(255),
    status VARCHAR(255) NOT NULL, audit_report_json TEXT, snapshot_object_name VARCHAR(255), uploaded_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS finetune_jobs (
    id BIGSERIAL PRIMARY KEY, job_uuid VARCHAR(64) NOT NULL UNIQUE, user_id BIGINT NOT NULL REFERENCES users(id),
    dataset_id BIGINT REFERENCES datasets(id), base_model_id VARCHAR(255) NOT NULL, recipe VARCHAR(255) NOT NULL,
    model_params_b VARCHAR(255), dataset_minio_uri VARCHAR(255), export_hf_repo VARCHAR(255), export_format VARCHAR(255),
    eval_prompts_json TEXT, status VARCHAR(255) NOT NULL, report_json TEXT, error_message TEXT, attempt_id VARCHAR(64),
    last_event_seq BIGINT NOT NULL DEFAULT 0, lease_expires_at TIMESTAMP, created_at TIMESTAMP, updated_at TIMESTAMP
);
