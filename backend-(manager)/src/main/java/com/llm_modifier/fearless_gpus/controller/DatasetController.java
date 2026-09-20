package com.llm_modifier.fearless_gpus.controller;

import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import com.llm_modifier.fearless_gpus.repository.FinetuneJobRepository;
import com.llm_modifier.fearless_gpus.service.MinioService;
import com.llm_modifier.fearless_gpus.service.RabbitMQProducer;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import com.llm_modifier.fearless_gpus.dto.HuggingFaceRequest;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

@RestController
@RequestMapping("/api/datasets")
@RequiredArgsConstructor
public class DatasetController {

    private final MinioService minioService;
    private final DatasetRecordRepository datasetRepository;
    private final RabbitMQProducer rabbitMQProducer;
    private final FinetuneJobRepository finetuneJobRepository;

    @PostMapping("/upload")
    public ResponseEntity<DatasetRecord> uploadDataset(
            @RequestParam("file") MultipartFile file,
            @AuthenticationPrincipal User user) {
        
        String objectName = minioService.uploadFile(file);
        
        DatasetRecord record = DatasetRecord.builder()
                .user(user)
                .filename(file.getOriginalFilename())
                .minioObjectName(objectName)
                .status("PENDING")
                .build();
                
        datasetRepository.save(record);
        
        rabbitMQProducer.sendDatasetForProcessing(record.getId(), objectName);
        
        return ResponseEntity.ok(record);
    }

    @PostMapping("/huggingface")
    public ResponseEntity<DatasetRecord> importHuggingFace(
            @RequestBody HuggingFaceRequest request,
            @AuthenticationPrincipal User user) {
        
        DatasetRecord record = DatasetRecord.builder()
                .user(user)
                .filename(request.getHuggingFaceId())
                .source("HUGGINGFACE")
                .huggingFaceId(request.getHuggingFaceId())
                .huggingFaceConfig(request.getHuggingFaceConfig())
                .minioObjectName("") // Bypass Postgres NOT NULL constraint from old schema
                .status("PENDING")
                .build();
                
        datasetRepository.save(record);
        
        rabbitMQProducer.sendHuggingFaceDatasetForProcessing(
                record.getId(),
                request.getHuggingFaceId(),
                request.getHuggingFaceConfig());
        
        return ResponseEntity.ok(record);
    }

    @GetMapping
    public ResponseEntity<List<DatasetRecord>> getDatasets(@AuthenticationPrincipal User user) {
        if (user.getRole().name().equals("ROLE_ADMIN")) {
            return ResponseEntity.ok(datasetRepository.findAll());
        }
        return ResponseEntity.ok(datasetRepository.findByUserId(user.getId()));
    }
    
    @GetMapping("/{id}/audit")
    public ResponseEntity<String> getAuditReport(@PathVariable Long id, @AuthenticationPrincipal User user) {
        DatasetRecord record = datasetRepository.findById(id).orElseThrow();
        if (user.getRole().name().equals("ROLE_ADMIN") || record.getUser().getId().equals(user.getId())) {
            return ResponseEntity.ok(record.getAuditReportJson());
        }
        return ResponseEntity.status(403).build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteDataset(
            @PathVariable Long id,
            @AuthenticationPrincipal User user) {
        DatasetRecord record = datasetRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Dataset not found"));
        boolean admin = user.getRole().name().equals("ROLE_ADMIN");
        if (!admin && !record.getUser().getId().equals(user.getId())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Dataset not owned by user");
        }
        if (finetuneJobRepository.existsByDatasetId(id)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Delete jobs using this dataset first");
        }

        minioService.deleteFile(record.getMinioObjectName());
        datasetRepository.delete(record);
        return ResponseEntity.noContent().build();
    }
}
