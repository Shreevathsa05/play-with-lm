package com.llm_modifier.fearless_gpus.controller;

import com.llm_modifier.fearless_gpus.service.HuggingFaceCatalogService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/huggingface")
@RequiredArgsConstructor
public class HuggingFaceCatalogController {

    private final HuggingFaceCatalogService catalogService;

    @GetMapping("/models")
    public ResponseEntity<List<Map<String, Object>>> listModels(
            @RequestParam(defaultValue = "") String q,
            @RequestParam(defaultValue = "12") int limit) {
        return ResponseEntity.ok(catalogService.listUnslothModels(q, limit));
    }
}
