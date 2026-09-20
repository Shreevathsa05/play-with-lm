package com.llm_modifier.fearless_gpus.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class HuggingFaceRequest {
    private String huggingFaceId;
    /** Optional builder config, e.g. SFT for smoltalk2. */
    private String huggingFaceConfig;
}
