package com.llm_modifier.fearless_gpus.service;

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.util.List;
import java.util.Map;

@Service
public class HuggingFaceCatalogService {

    private final RestClient restClient = RestClient.create();

    public List<Map<String, Object>> listUnslothModels(String query, int requestedLimit) {
        int limit = Math.max(1, Math.min(requestedLimit, 30));
        UriComponentsBuilder builder = UriComponentsBuilder
                .fromUriString("https://huggingface.co/api/models")
                .queryParam("author", "unsloth")
                .queryParam("sort", "downloads")
                .queryParam("direction", -1)
                .queryParam("limit", limit)
                .queryParam("expand", "safetensors")
                .queryParam("expand", "downloads")
                .queryParam("expand", "likes")
                .queryParam("expand", "pipeline_tag")
                .queryParam("expand", "gated")
                .queryParam("expand", "lastModified");
        if (query != null && !query.isBlank()) {
            builder.queryParam("search", query.trim());
        }
        URI uri = builder.build().encode().toUri();
        List<Map<String, Object>> result = restClient.get()
                .uri(uri)
                .retrieve()
                .body(new ParameterizedTypeReference<>() {});
        return result == null ? List.of() : result;
    }
}
