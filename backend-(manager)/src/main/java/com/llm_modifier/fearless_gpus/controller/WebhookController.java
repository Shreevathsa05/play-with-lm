package com.llm_modifier.fearless_gpus.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.llm_modifier.fearless_gpus.dto.JobWebhookRequest;
import com.llm_modifier.fearless_gpus.dto.WebhookRequest;
import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import com.llm_modifier.fearless_gpus.service.FinetuneJobService;
import org.springframework.http.ResponseEntity;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Map;

@RestController
@RequestMapping("/api/webhook")
public class WebhookController {

    private final DatasetRecordRepository datasetRecordRepository;
    private final FinetuneJobService jobService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${worker.internal-token:}")
    private String workerToken;

    public WebhookController(DatasetRecordRepository datasetRecordRepository, FinetuneJobService jobService) {
        this.datasetRecordRepository = datasetRecordRepository;
        this.jobService = jobService;
    }

    @PostMapping("/datasets/{id}")
    public ResponseEntity<Void> handleDatasetWebhook(@PathVariable Long id,
                                                      @RequestHeader(value = "X-Worker-Token", required = false) String token,
                                                      @RequestBody WebhookRequest request) {
        if (!authorized(token)) return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        return datasetRecordRepository.findById(id).map(dataset -> {
            dataset.setStatus(request.getStatus());
            try {
                if (request.getAuditReport() != null) {
                    dataset.setAuditReportJson(objectMapper.writeValueAsString(request.getAuditReport()));
                    java.util.Map<String, Object> audit = objectMapper.convertValue(
                            request.getAuditReport(), new com.fasterxml.jackson.core.type.TypeReference<java.util.Map<String, Object>>() {});
                    Object snapshot = audit.get("snapshot");
                    if (snapshot instanceof java.util.Map<?, ?> snapshotMap
                            && snapshotMap.get("object_name") != null) {
                        dataset.setSnapshotObjectName(String.valueOf(snapshotMap.get("object_name")));
                    }
                }
            } catch (JsonProcessingException e) {
                dataset.setAuditReportJson("{\"error\": \"Failed to serialize audit report\"}");
            }
            datasetRecordRepository.save(dataset);
            return ResponseEntity.ok().<Void>build();
        }).orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/jobs/{jobUuid}")
    public ResponseEntity<Void> handleJobWebhook(
            @PathVariable String jobUuid,
            @RequestHeader(value = "X-Worker-Token", required = false) String token,
            @RequestBody JobWebhookRequest request) {
        if (!authorized(token)) return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        jobService.handleWebhook(jobUuid, request);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/jobs/{jobUuid}/control")
    public ResponseEntity<Map<String, Boolean>> jobControl(
            @PathVariable String jobUuid,
            @RequestHeader(value = "X-Worker-Token", required = false) String token) {
        if (!authorized(token)) return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        return ResponseEntity.ok(Map.of("cancel_requested", jobService.isCancellationRequested(jobUuid)));
    }

    private boolean authorized(String candidate) {
        if (workerToken == null || workerToken.isBlank() || candidate == null) return false;
        return MessageDigest.isEqual(workerToken.getBytes(StandardCharsets.UTF_8),
                candidate.getBytes(StandardCharsets.UTF_8));
    }
}
