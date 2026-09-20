package com.llm_modifier.fearless_gpus;

import com.llm_modifier.fearless_gpus.config.RabbitMQConfig;
import com.llm_modifier.fearless_gpus.dto.CreateJobRequest;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Lightweight tests that do not require Postgres / RabbitMQ / MinIO.
 * Full @SpringBootTest contextLoads needs docker-compose infra.
 */
class FearlessGpusApplicationTests {

    @Test
    void finetuneQueueContract() {
        assertEquals("finetuning.jobs.queue", RabbitMQConfig.FINETUNE_QUEUE_NAME);
        assertEquals("finetuning.exchange", RabbitMQConfig.FINETUNE_EXCHANGE_NAME);
        assertEquals("finetuning.routing.key", RabbitMQConfig.FINETUNE_ROUTING_KEY);
    }

    @Test
    void createJobRequestHoldsEvalCapShape() {
        CreateJobRequest req = new CreateJobRequest();
        req.setBaseModelId("google/gemma-3-270m-it");
        req.setRecipe("qlora_4bit");
        req.setEvalPrompts(List.of(Map.of("prompt", "hi", "reference", "hi")));
        req.setDryRun(true);
        assertEquals(1, req.getEvalPrompts().size());
        assertTrue(req.getDryRun());
        assertTrue(Set.of("fullparams", "lora", "qlora_4bit", "qlora_8bit", "embedding")
                .contains(req.getRecipe()));
    }
}
