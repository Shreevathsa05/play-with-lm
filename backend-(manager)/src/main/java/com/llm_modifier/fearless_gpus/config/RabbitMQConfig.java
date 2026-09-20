package com.llm_modifier.fearless_gpus.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMQConfig {

    // Dataset processing
    public static final String QUEUE_NAME = "dataset.processing.queue";
    public static final String EXCHANGE_NAME = "dataset.exchange";
    public static final String ROUTING_KEY = "dataset.routing.key";

    // Fine-tuning jobs
    public static final String FINETUNE_QUEUE_NAME = "finetuning.jobs.queue";
    public static final String FINETUNE_EXCHANGE_NAME = "finetuning.exchange";
    public static final String FINETUNE_ROUTING_KEY = "finetuning.routing.key";

    @Bean
    public Queue queue() {
        return new Queue(QUEUE_NAME, true);
    }

    @Bean
    public DirectExchange exchange() {
        return new DirectExchange(EXCHANGE_NAME);
    }

    @Bean
    public Binding binding(Queue queue, DirectExchange exchange) {
        return BindingBuilder.bind(queue).to(exchange).with(ROUTING_KEY);
    }

    @Bean
    public Queue finetuneQueue() {
        return new Queue(FINETUNE_QUEUE_NAME, true);
    }

    @Bean
    public DirectExchange finetuneExchange() {
        return new DirectExchange(FINETUNE_EXCHANGE_NAME);
    }

    @Bean
    public Binding finetuneBinding(Queue finetuneQueue, DirectExchange finetuneExchange) {
        return BindingBuilder.bind(finetuneQueue).to(finetuneExchange).with(FINETUNE_ROUTING_KEY);
    }

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        final RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(jsonMessageConverter());
        return rabbitTemplate;
    }
}
