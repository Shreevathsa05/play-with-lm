package com.llm_modifier.fearless_gpus.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class JobResponse {
    private Long id;
    private String jobUuid;
    private Long datasetId;
    private String baseModelId;
    private String recipe;
    private String objective;
    private String resumeFromCheckpoint;
    private String modelParamsB;
    private String datasetMinioUri;
    private String exportHfRepo;
    private String exportFormat;
    private String status;
    private String reportJson;
    private String errorMessage;
    private String userEmail;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
