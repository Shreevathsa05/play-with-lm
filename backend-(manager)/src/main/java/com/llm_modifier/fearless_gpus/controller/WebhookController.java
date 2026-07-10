package com.llm_modifier.fearless_gpus.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.llm_modifier.fearless_gpus.dto.WebhookRequest;
import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/webhook/datasets")
public class WebhookController {

    private final DatasetRecordRepository datasetRecordRepository;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public WebhookController(DatasetRecordRepository datasetRecordRepository) {
        this.datasetRecordRepository = datasetRecordRepository;
    }

    @PostMapping("/{id}")
    public ResponseEntity<Void> handleWebhook(@PathVariable Long id, @RequestBody WebhookRequest request) {
        return datasetRecordRepository.findById(id).map(dataset -> {
            dataset.setStatus(request.getStatus());
            try {
                if (request.getAuditReport() != null) {
                    dataset.setAuditReportJson(objectMapper.writeValueAsString(request.getAuditReport()));
                }
            } catch (JsonProcessingException e) {
                // If it fails to serialize, we just save the error in the field
                dataset.setAuditReportJson("{\"error\": \"Failed to serialize audit report\"}");
            }
            datasetRecordRepository.save(dataset);
            return ResponseEntity.ok().<Void>build();
        }).orElse(ResponseEntity.notFound().build());
    }
}
