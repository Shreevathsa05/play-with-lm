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
    private Integer epochs;
    private Integer maxSteps;
    private Double learningRate;
    private Integer gradAccumSteps;
    private Double warmupRatio;
    /** Stop early when logged train loss is at or below this value. */
    private Double targetLoss;
    /** Stop early when loss has fallen by this fraction of the first logged loss (0.5 = 50%). */
    private Double minLossDropRatio;
    private Boolean dryRun;
}
