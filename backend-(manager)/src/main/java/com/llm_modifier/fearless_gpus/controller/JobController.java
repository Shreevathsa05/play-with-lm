package com.llm_modifier.fearless_gpus.controller;

import com.llm_modifier.fearless_gpus.dto.CreateJobRequest;
import com.llm_modifier.fearless_gpus.dto.JobResponse;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.service.FinetuneJobService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/jobs")
@RequiredArgsConstructor
public class JobController {

    private final FinetuneJobService jobService;

    @PostMapping
    public ResponseEntity<JobResponse> createJob(
            @RequestBody CreateJobRequest request,
            @AuthenticationPrincipal User user) {
        return ResponseEntity.accepted().body(jobService.createJob(request, user));
    }

    @GetMapping
    public ResponseEntity<List<JobResponse>> listJobs(@AuthenticationPrincipal User user) {
        return ResponseEntity.ok(jobService.listJobs(user));
    }

    @GetMapping("/{jobUuid}")
    public ResponseEntity<JobResponse> getJob(
            @PathVariable String jobUuid,
            @AuthenticationPrincipal User user) {
        return ResponseEntity.ok(jobService.getJob(jobUuid, user));
    }

    @GetMapping("/{jobUuid}/report")
    public ResponseEntity<String> getReport(
            @PathVariable String jobUuid,
            @AuthenticationPrincipal User user) {
        JobResponse job = jobService.getJob(jobUuid, user);
        if (job.getReportJson() == null) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.ok(job.getReportJson());
    }

    @DeleteMapping("/{jobUuid}")
    public ResponseEntity<Void> deleteJob(
            @PathVariable String jobUuid,
            @AuthenticationPrincipal User user) {
        jobService.deleteJob(jobUuid, user);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{jobUuid}/cancel")
    public ResponseEntity<JobResponse> cancelJob(
            @PathVariable String jobUuid,
            @AuthenticationPrincipal User user) {
        return ResponseEntity.ok(jobService.cancelJob(jobUuid, user));
    }

    @PostMapping("/{jobUuid}/resume")
    public ResponseEntity<JobResponse> resumeJob(
            @PathVariable String jobUuid,
            @AuthenticationPrincipal User user) {
        return ResponseEntity.accepted().body(jobService.resumeJob(jobUuid, user));
    }
}
