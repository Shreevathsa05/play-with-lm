package com.llm_modifier.fearless_gpus.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.llm_modifier.fearless_gpus.config.RabbitMQConfig;
import com.llm_modifier.fearless_gpus.dto.CreateJobRequest;
import com.llm_modifier.fearless_gpus.dto.JobResponse;
import com.llm_modifier.fearless_gpus.dto.JobWebhookRequest;
import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import com.llm_modifier.fearless_gpus.model.FinetuneJob;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import com.llm_modifier.fearless_gpus.repository.FinetuneJobRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class FinetuneJobService {

    private static final Set<String> ALLOWED_RECIPES = Set.of(
            "fullparams", "lora", "qlora_4bit", "qlora_8bit", "embedding"
    );
    private static final Set<String> ALLOWED_OBJECTIVES = Set.of(
            "pretrain", "cpt", "sft", "dpo", "kto", "reward_model", "ppo", "grpo", "embedding"
    );
    private static final Map<String, Set<String>> OBJECTIVE_SCHEMAS = Map.of(
            "pretrain", Set.of("raw_text"), "cpt", Set.of("raw_text"), "sft", Set.of("sft"),
            "dpo", Set.of("paired_preference"), "reward_model", Set.of("paired_preference"),
            "kto", Set.of("binary_preference"), "ppo", Set.of("prompt_only"), "grpo", Set.of("prompt_only"),
            "embedding", Set.of("embedding_pair", "embedding_triplet")
    );
    private static final Set<String> ACTIVE_STATUSES = Set.of("CHECKING", "WAITING", "RUNNING", "CANCEL_REQUESTED");
    private static final long ACTIVE_HEARTBEAT_GRACE_SECONDS = 60;

    private final FinetuneJobRepository jobRepository;
    private final DatasetRecordRepository datasetRepository;
    private final RabbitTemplate rabbitTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public JobResponse createJob(CreateJobRequest request, User user) {
        if (request.getBaseModelId() == null || request.getBaseModelId().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "baseModelId is required");
        }
        String recipe = request.getRecipe() == null ? "" : request.getRecipe().trim().toLowerCase();
        if (!ALLOWED_RECIPES.contains(recipe)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "Invalid recipe. Allowed: " + ALLOWED_RECIPES);
        }
        if (request.getObjective() != null && !request.getObjective().isBlank()
                && !ALLOWED_OBJECTIVES.contains(request.getObjective().trim().toLowerCase())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid training objective");
        }
        if ("ppo".equalsIgnoreCase(request.getObjective())
                && (request.getRewardModelId() == null || request.getRewardModelId().isBlank())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "PPO requires rewardModelId");
        }

        DatasetRecord dataset = null;
        String datasetMinioUri = request.getDatasetMinioUri();

        if (request.getDatasetId() == null && datasetMinioUri != null && !datasetMinioUri.isBlank()
                && !user.getRole().name().equals("ROLE_ADMIN")) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN,
                    "Select an owned prepared dataset instead of supplying a storage URI");
        }

        if (request.getDatasetId() != null) {
            dataset = datasetRepository.findById(request.getDatasetId())
                    .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Dataset not found"));
            if (!user.getRole().name().equals("ROLE_ADMIN")
                    && !dataset.getUser().getId().equals(user.getId())) {
                throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Dataset not owned by user");
            }
            if (!"COMPLETED".equalsIgnoreCase(dataset.getStatus())) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                        "Dataset must be COMPLETED before starting a fine-tune job");
            }
            if (dataset.getSnapshotObjectName() == null || dataset.getSnapshotObjectName().isBlank()) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                        "Dataset has no immutable prepared snapshot; prepare it again before training");
            }
            String requestedObjective = request.getObjective() == null ? "" : request.getObjective().trim().toLowerCase();
            if (!requestedObjective.isBlank() && dataset.getAuditReportJson() != null) {
                try {
                    Map<?, ?> report = objectMapper.readValue(dataset.getAuditReportJson(), Map.class);
                    String schema = String.valueOf(report.get("schema_type"));
                    if (!OBJECTIVE_SCHEMAS.get(requestedObjective).contains(schema)) {
                        throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                                "Objective '" + requestedObjective + "' requires a different prepared dataset schema");
                    }
                } catch (JsonProcessingException e) {
                    throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Dataset audit report is unreadable; prepare it again");
                }
            }
            if (datasetMinioUri == null || datasetMinioUri.isBlank()) {
                if (dataset.getSnapshotObjectName() != null && !dataset.getSnapshotObjectName().isBlank()) {
                    datasetMinioUri = "minio://datasets/" + dataset.getSnapshotObjectName();
                } else if (dataset.getMinioObjectName() != null && !dataset.getMinioObjectName().isBlank()) {
                    datasetMinioUri = "minio://datasets/" + dataset.getMinioObjectName();
                }
            }
        }

        if ((datasetMinioUri == null || datasetMinioUri.isBlank())
                && (dataset == null || dataset.getHuggingFaceId() == null)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "Provide datasetId with MinIO object or datasetMinioUri");
        }

        String jobUuid = UUID.randomUUID().toString();
        String evalJson = null;
        try {
            if (request.getEvalPrompts() != null) {
                if (request.getEvalPrompts().size() > 1000) {
                    throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                            "evalPrompts capped at 1000");
                }
                evalJson = objectMapper.writeValueAsString(request.getEvalPrompts());
            }
        } catch (JsonProcessingException e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid evalPrompts JSON");
        }

        FinetuneJob job = FinetuneJob.builder()
                .jobUuid(jobUuid)
                .user(user)
                .dataset(dataset)
                .baseModelId(request.getBaseModelId().trim())
                .recipe(recipe)
                .objective(resolveObjective(dataset, request.getObjective()))
                .rewardModelId(request.getRewardModelId())
                .modelParamsB(request.getModelParamsB())
                .datasetMinioUri(datasetMinioUri)
                .exportHfRepo(request.getExportHfRepo())
                .exportFormat(resolveExportFormat(recipe, request.getExportFormat()))
                .evalPromptsJson(evalJson)
                .status("PENDING")
                .build();

        jobRepository.save(job);

        Map<String, Object> payload = buildQueuePayload(job, request, dataset);
        rabbitTemplate.convertAndSend(
                RabbitMQConfig.FINETUNE_EXCHANGE_NAME,
                RabbitMQConfig.FINETUNE_ROUTING_KEY,
                payload
        );
        log.info("Enqueued finetune job {}", jobUuid);

        job.setStatus("QUEUED");
        jobRepository.save(job);
        return toResponse(job);
    }

    private String resolveExportFormat(String recipe, String requested) {
        if (requested != null && !requested.isBlank()) {
            return requested.trim().toLowerCase();
        }
        return "fullparams".equals(recipe) ? "merged_16bit" : "lora";
    }

    private String resolveObjective(DatasetRecord dataset, String requested) {
        if (requested != null && !requested.isBlank()) {
            return requested.trim().toLowerCase();
        }
        if (dataset == null || dataset.getAuditReportJson() == null) return "sft";
        try {
            Map<?, ?> report = objectMapper.readValue(dataset.getAuditReportJson(), Map.class);
            String schema = String.valueOf(report.get("schema_type"));
            return switch (schema) {
                case "raw_text" -> "cpt";
                case "paired_preference" -> "dpo";
                case "binary_preference" -> "kto";
                case "embedding_pair", "embedding_triplet" -> "embedding";
                default -> "sft";
            };
        } catch (JsonProcessingException ignored) {
            return "sft";
        }
    }

    private Map<String, Object> buildQueuePayload(FinetuneJob job, CreateJobRequest request, DatasetRecord dataset) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("job_uuid", job.getJobUuid());
        payload.put("base_model_id", job.getBaseModelId());
        payload.put("recipe", job.getRecipe());
        if (job.getModelParamsB() != null) {
            payload.put("model_params_b", job.getModelParamsB());
        }
        if (job.getDatasetMinioUri() != null) {
            payload.put("dataset_minio_uri", job.getDatasetMinioUri());
        }
        if (dataset != null && "HUGGINGFACE".equalsIgnoreCase(dataset.getSource())
                && dataset.getHuggingFaceId() != null
                && (job.getDatasetMinioUri() == null || job.getDatasetMinioUri().isBlank())) {
            Map<String, Object> hfDataset = new HashMap<>();
            hfDataset.put("path", dataset.getHuggingFaceId());
            hfDataset.put("split", "train");
            payload.put("hf_dataset", hfDataset);
        }

        Map<String, Object> export = new HashMap<>();
        export.put("format", job.getExportFormat());
        export.put("minio_destination", "minio://jobs/" + job.getJobUuid() + "/exports");
        if (job.getExportHfRepo() != null && !job.getExportHfRepo().isBlank()) {
            export.put("hf_repo", job.getExportHfRepo());
        }
        // Secrets never enter RabbitMQ. The worker resolves HF_TOKEN from its environment.
        export.put("private", true);
        payload.put("export", export);

        Map<String, Object> eval = new HashMap<>();
        if (request.getEvalPrompts() != null) {
            eval.put("prompts", request.getEvalPrompts());
        }
        eval.put("max_prompts", 1000);
        payload.put("eval", eval);

        payload.put("objective", job.getObjective());
        if (dataset != null && dataset.getAuditReportJson() != null) {
            try {
                Map<?, ?> report = objectMapper.readValue(dataset.getAuditReportJson(), Map.class);
                payload.put("dataset_schema", String.valueOf(report.get("schema_type")));
            } catch (JsonProcessingException ignored) { }
        }
        if (job.getRewardModelId() != null && !job.getRewardModelId().isBlank()) {
            payload.put("reward_model_id", job.getRewardModelId().trim());
        }

        if (request.getMaxSeqLength() != null) {
            payload.put("max_seq_length", request.getMaxSeqLength());
        }
        if (request.getBatchSize() != null) {
            payload.put("batch_size", request.getBatchSize());
        }

        Map<String, Object> train = new HashMap<>();
        if (request.getBatchSize() != null) {
            train.put("batch_size", request.getBatchSize());
        }
        if (request.getEpochs() != null) {
            train.put("epochs", request.getEpochs());
        }
        if (request.getMaxSteps() != null) {
            train.put("max_steps", request.getMaxSteps());
        }
        if (request.getLearningRate() != null) {
            train.put("learning_rate", request.getLearningRate());
        }
        if (request.getGradAccumSteps() != null) {
            train.put("grad_accum_steps", request.getGradAccumSteps());
        }
        if (request.getWarmupRatio() != null) {
            train.put("warmup_ratio", request.getWarmupRatio());
        }
        if (request.getTargetLoss() != null) {
            train.put("target_loss", request.getTargetLoss());
        }
        if (request.getMinLossDropRatio() != null) {
            train.put("min_loss_drop_ratio", request.getMinLossDropRatio());
        }
        if (!train.isEmpty()) {
            payload.put("train", train);
        }

        if (Boolean.TRUE.equals(request.getDryRun())) {
            payload.put("dry_run", true);
        }
        payload.put("unsloth_supported", true);
        return payload;
    }

    public List<JobResponse> listJobs(User user) {
        if (user.getRole().name().equals("ROLE_ADMIN")) {
            return jobRepository.findAll().stream().map(this::toResponse).toList();
        }
        return jobRepository.findByUserIdOrderByCreatedAtDesc(user.getId()).stream()
                .map(this::toResponse)
                .toList();
    }

    public JobResponse getJob(String jobUuid, User user) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        if (!user.getRole().name().equals("ROLE_ADMIN")
                && !job.getUser().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Not allowed");
        }
        return toResponse(job);
    }

    public JobResponse cancelJob(String jobUuid, User user) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        if (!user.getRole().name().equals("ROLE_ADMIN") && !job.getUser().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Job not owned by user");
        }
        String current = job.getStatus().toUpperCase(Locale.ROOT);
        if (Set.of("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED").contains(current)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Terminal jobs cannot be cancelled");
        }
        job.setStatus(Set.of("PENDING", "QUEUED").contains(current) ? "CANCELLED" : "CANCEL_REQUESTED");
        jobRepository.save(job);
        return toResponse(job);
    }

    public JobResponse resumeJob(String jobUuid, User user) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        if (!user.getRole().name().equals("ROLE_ADMIN") && !job.getUser().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Job not owned by user");
        }
        if (!Set.of("CANCELLED", "INTERRUPTED").contains(job.getStatus().toUpperCase(Locale.ROOT))) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Only cancelled or interrupted jobs can resume");
        }
        String checkpoint = job.getResumeFromCheckpoint();
        if (checkpoint == null || checkpoint.isBlank()) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "No saved checkpoint is available to resume");
        }
        job.setAttemptId(null);
        job.setLastEventSeq(0);
        job.setLeaseExpiresAt(null);
        job.setStatus("PENDING");
        jobRepository.save(job);
        DatasetRecord dataset = job.getDataset() == null ? null : datasetRepository.findById(job.getDataset().getId()).orElse(null);
        Map<String, Object> payload = buildQueuePayload(job, new CreateJobRequest(), dataset);
        payload.put("resume_from_checkpoint", checkpoint);
        rabbitTemplate.convertAndSend(RabbitMQConfig.FINETUNE_EXCHANGE_NAME, RabbitMQConfig.FINETUNE_ROUTING_KEY, payload);
        job.setStatus("QUEUED");
        jobRepository.save(job);
        return toResponse(job);
    }

    public boolean isCancellationRequested(String jobUuid) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        return "CANCEL_REQUESTED".equalsIgnoreCase(job.getStatus()) || "CANCELLED".equalsIgnoreCase(job.getStatus());
    }

    public void deleteJob(String jobUuid, User user) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        if (!user.getRole().name().equals("ROLE_ADMIN")
                && !job.getUser().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Job not owned by user");
        }
        String status = job.getStatus().toUpperCase(Locale.ROOT);
        boolean liveHeartbeat = ACTIVE_STATUSES.contains(status)
                && job.getUpdatedAt() != null
                && job.getUpdatedAt().isAfter(LocalDateTime.now().minusSeconds(ACTIVE_HEARTBEAT_GRACE_SECONDS));
        if (liveHeartbeat) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "This job still has a live worker heartbeat; wait for it to finish or stop updating");
        }
        jobRepository.delete(job);
    }

    public void handleWebhook(String jobUuid, JobWebhookRequest request) {
        FinetuneJob job = jobRepository.findByJobUuid(jobUuid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Job not found"));
        // A redelivered message must not start a second training process while
        // the first attempt is still running (or after it has finished).
        if ("CHECKING".equalsIgnoreCase(request.getStatus()) && job.getAttemptId() != null) {
            Map<String, Object> claim = objectMapper.convertValue(request.getReport(),
                    new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {});
            Object incomingAttempt = claim == null ? null : claim.get("attempt_id");
            if (!job.getAttemptId().equals(String.valueOf(incomingAttempt))) {
                throw new ResponseStatusException(HttpStatus.CONFLICT,
                        "Job has already been claimed by another worker attempt");
            }
        }
        // Heartbeats may repeat the same status/report. Touch the timestamp
        // explicitly so Hibernate still persists proof that the worker is alive.
        job.setUpdatedAt(LocalDateTime.now());
        if (Set.of("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED").contains(job.getStatus())
                && request.getStatus() != null
                && !job.getStatus().equalsIgnoreCase(request.getStatus())) {
            log.warn("Ignoring status regression {} -> {} for job {}", job.getStatus(), request.getStatus(), jobUuid);
            return;
        }
        if (request.getStatus() != null) {
            job.setStatus(request.getStatus());
        }
        try {
            if (request.getReport() != null) {
                Map<String, Object> incoming = objectMapper.convertValue(
                        request.getReport(), new com.fasterxml.jackson.core.type.TypeReference<LinkedHashMap<String, Object>>() {});
                Object attempt = incoming.get("attempt_id");
                Object sequence = incoming.get("event_seq");
                if (attempt != null && sequence instanceof Number seq) {
                    long incomingSeq = seq.longValue();
                    if (job.getAttemptId() != null && !job.getAttemptId().equals(String.valueOf(attempt))) {
                        log.warn("Ignoring webhook from stale attempt {} for job {}", attempt, jobUuid);
                        return;
                    }
                    if (incomingSeq <= job.getLastEventSeq()) {
                        return;
                    }
                    if (job.getAttemptId() == null) {
                        job.setAttemptId(String.valueOf(attempt));
                    }
                    job.setLastEventSeq(incomingSeq);
                    job.setLeaseExpiresAt(LocalDateTime.now().plusSeconds(ACTIVE_HEARTBEAT_GRACE_SECONDS));
                }
                Map<String, Object> merged = mergeReports(parseReport(job.getReportJson()), incoming);
                job.setReportJson(objectMapper.writeValueAsString(merged));
                Object checkpoint = incoming.get("resume_from_checkpoint");
                if (checkpoint != null && !String.valueOf(checkpoint).isBlank()) {
                    job.setResumeFromCheckpoint(String.valueOf(checkpoint));
                }
                Object reason = merged.get("reason");
                if (reason == null) {
                    reason = merged.get("error");
                }
                if (reason == null) {
                    reason = merged.get("stderr_tail");
                }
                if (reason != null) {
                    job.setErrorMessage(String.valueOf(reason));
                }
            }
            if (request.getErrorMessage() != null) {
                job.setErrorMessage(request.getErrorMessage());
            }
        } catch (JsonProcessingException e) {
            job.setReportJson("{\"error\":\"Failed to serialize report\"}");
        }
        jobRepository.save(job);
        log.info("Job {} webhook status={}", jobUuid, job.getStatus());
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseReport(String reportJson) throws JsonProcessingException {
        if (reportJson == null || reportJson.isBlank()) {
            return new LinkedHashMap<>();
        }
        return objectMapper.readValue(reportJson, LinkedHashMap.class);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> mergeReports(Map<String, Object> existing, Map<String, Object> incoming) {
        Map<String, Object> merged = new LinkedHashMap<>(existing);
        for (Map.Entry<String, Object> entry : incoming.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            if ("event".equals(key) && value instanceof Map<?, ?> eventMap) {
                appendEvent(merged, (Map<String, Object>) eventMap);
                continue;
            }
            if ("events".equals(key) && value instanceof List<?> events) {
                for (Object item : events) {
                    if (item instanceof Map<?, ?> eventMap) {
                        appendEvent(merged, (Map<String, Object>) eventMap);
                    }
                }
                continue;
            }
            if (value instanceof Map<?, ?> incomingMap && merged.get(key) instanceof Map<?, ?> existingMap) {
                merged.put(key, mergeReports((Map<String, Object>) existingMap, (Map<String, Object>) incomingMap));
            } else {
                merged.put(key, value);
            }
        }
        return merged;
    }

    @SuppressWarnings("unchecked")
    private void appendEvent(Map<String, Object> report, Map<String, Object> event) {
        List<Map<String, Object>> events = report.containsKey("events")
                && report.get("events") instanceof List<?> list
                ? new ArrayList<>((List<Map<String, Object>>) list)
                : new ArrayList<>();
        events.add(new LinkedHashMap<>(event));
        int maxEvents = 500;
        if (events.size() > maxEvents) {
            events = new ArrayList<>(events.subList(events.size() - maxEvents, events.size()));
        }
        report.put("events", events);
    }

    private JobResponse toResponse(FinetuneJob job) {
        return JobResponse.builder()
                .id(job.getId())
                .jobUuid(job.getJobUuid())
                .datasetId(job.getDataset() != null ? job.getDataset().getId() : null)
                .baseModelId(job.getBaseModelId())
                .recipe(job.getRecipe())
                .objective(job.getObjective())
                .resumeFromCheckpoint(job.getResumeFromCheckpoint())
                .modelParamsB(job.getModelParamsB())
                .datasetMinioUri(job.getDatasetMinioUri())
                .exportHfRepo(job.getExportHfRepo())
                .exportFormat(job.getExportFormat())
                .status(job.getStatus())
                .reportJson(job.getReportJson())
                .errorMessage(job.getErrorMessage())
                .userEmail(job.getUser() != null ? job.getUser().getEmail() : null)
                .createdAt(job.getCreatedAt())
                .updatedAt(job.getUpdatedAt())
                .build();
    }
}
