package com.llm_modifier.fearless_gpus.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreateJobRequest {
    private Long datasetId;
    private String baseModelId;
    private String recipe;
    private String objective;
    private String rewardModelId;
    private String modelParamsB;
    private String datasetMinioUri;
    private String exportHfRepo;
    private String exportFormat;
    private String hfToken;
    private List<Map<String, Object>> evalPrompts;
    private Integer maxSeqLength;
    private Integer batchSize;
    private Boolean dryRun;
}
