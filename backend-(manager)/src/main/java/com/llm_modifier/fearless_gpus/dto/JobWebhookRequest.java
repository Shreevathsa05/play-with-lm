package com.llm_modifier.fearless_gpus.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class JobWebhookRequest {
    private String status;
    private Object report;
    private String errorMessage;
}
