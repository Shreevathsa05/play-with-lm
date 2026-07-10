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

    public void sendHuggingFaceDatasetForProcessing(Long datasetId, String huggingFaceId) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("datasetId", datasetId);
        payload.put("source", "HUGGINGFACE");
        payload.put("huggingFaceId", huggingFaceId);
        log.info("Sending HuggingFace processing message: {}", payload);
        rabbitTemplate.convertAndSend(RabbitMQConfig.EXCHANGE_NAME, RabbitMQConfig.ROUTING_KEY, payload);
    }
}
