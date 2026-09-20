package com.llm_modifier.fearless_gpus;

import com.llm_modifier.fearless_gpus.config.RabbitMQConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FinetuneQueueConfigTest {

    @Test
    void finetuneQueueConstantsMatchWorker() {
        assertEquals("finetuning.jobs.queue", RabbitMQConfig.FINETUNE_QUEUE_NAME);
        assertEquals("finetuning.exchange", RabbitMQConfig.FINETUNE_EXCHANGE_NAME);
        assertEquals("finetuning.routing.key", RabbitMQConfig.FINETUNE_ROUTING_KEY);
        assertEquals("dataset.processing.queue", RabbitMQConfig.QUEUE_NAME);
    }
}
