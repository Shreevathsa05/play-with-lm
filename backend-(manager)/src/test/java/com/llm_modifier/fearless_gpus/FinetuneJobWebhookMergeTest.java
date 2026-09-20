package com.llm_modifier.fearless_gpus;

import com.llm_modifier.fearless_gpus.dto.JobWebhookRequest;
import com.llm_modifier.fearless_gpus.model.FinetuneJob;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import com.llm_modifier.fearless_gpus.repository.FinetuneJobRepository;
import com.llm_modifier.fearless_gpus.service.FinetuneJobService;
import org.junit.jupiter.api.Test;
import org.springframework.amqp.rabbit.core.RabbitTemplate;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FinetuneJobWebhookMergeTest {

    @Test
    void handleWebhook_mergesEventsAndDeepNestedReports() throws Exception {
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        DatasetRecordRepository datasets = mock(DatasetRecordRepository.class);
        RabbitTemplate rabbit = mock(RabbitTemplate.class);
        FinetuneJobService service = new FinetuneJobService(jobs, datasets, rabbit);

        FinetuneJob job = FinetuneJob.builder()
                .jobUuid("job-1")
                .user(User.builder().id(1L).email("student@test").build())
                .status("RUNNING")
                .reportJson("{\"phase\":\"old\",\"export\":{\"hf_push\":{\"status\":\"pushing\",\"repo_id\":\"u/m\"}},\"events\":[{\"event\":\"dataset_loaded\"}]}")
                .build();
        when(jobs.findByJobUuid("job-1")).thenReturn(Optional.of(job));
        when(jobs.save(any(FinetuneJob.class))).thenAnswer(invocation -> invocation.getArgument(0));

        Map<String, Object> incoming = new LinkedHashMap<>();
        incoming.put("phase", "Training in progress");
        incoming.put("event", Map.of("event", "train_progress", "step", 2, "total", 10));
        Map<String, Object> hfPush = new LinkedHashMap<>();
        hfPush.put("status", "pushing");
        hfPush.put("repo_id", "u/m");
        hfPush.put("url", null);
        incoming.put("export", Map.of("hf_push", hfPush));

        JobWebhookRequest request = new JobWebhookRequest();
        request.setStatus("RUNNING");
        request.setReport(incoming);
        service.handleWebhook("job-1", request);

        Map<?, ?> report = new com.fasterxml.jackson.databind.ObjectMapper().readValue(job.getReportJson(), Map.class);
        assertEquals("Training in progress", report.get("phase"));
        assertTrue(report.get("events") instanceof List<?> events && events.size() == 2);
        Map<?, ?> export = (Map<?, ?>) report.get("export");
        Map<?, ?> mergedHfPush = (Map<?, ?>) export.get("hf_push");
        assertEquals("pushing", mergedHfPush.get("status"));
        assertEquals("u/m", mergedHfPush.get("repo_id"));
    }
}
