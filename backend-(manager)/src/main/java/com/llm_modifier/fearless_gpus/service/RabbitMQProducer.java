package com.llm_modifier.fearless_gpus.service;

import com.llm_modifier.fearless_gpus.config.RabbitMQConfig;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class RabbitMQProducer {

    private final RabbitTemplate rabbitTemplate;

    public void sendDatasetForProcessing(Long datasetId, String minioObjectName) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("datasetId", datasetId);
        payload.put("source", "MINIO");
        payload.put("minioObjectName", minioObjectName);
        log.info("Sending dataset processing message: {}", payload);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY, payload);
    }

    public void sendHuggingFaceDatasetForProcessing(Long datasetId, String huggingFaceId, String huggingFaceConfig) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("datasetId", datasetId);
        payload.put("source", "HUGGINGFACE");
        payload.put("huggingFaceId", huggingFaceId);
        if (huggingFaceConfig != null && !huggingFaceConfig.isBlank()) {
            payload.put("huggingFaceConfig", huggingFaceConfig.trim());
        }
        log.info("Sending HuggingFace processing message: {}", payload);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY, payload);
    }

    public void sendFinetuneJob(Map<String, Object> payload) {
        log.info("Sending finetune job message for {}", payload.get("job_uuid"));
        rabbitTemplate.convertAndSend(
                RabbitMQConfig.FINETUNE_EXCHANGE_NAME,
                RabbitMQConfig.FINETUNE_ROUTING_KEY,
                payload
        );
    }
}
