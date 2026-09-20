package com.llm_modifier.fearless_gpus;

import com.llm_modifier.fearless_gpus.controller.DatasetController;
import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import com.llm_modifier.fearless_gpus.model.FinetuneJob;
import com.llm_modifier.fearless_gpus.model.Role;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.repository.DatasetRecordRepository;
import com.llm_modifier.fearless_gpus.repository.FinetuneJobRepository;
import com.llm_modifier.fearless_gpus.service.FinetuneJobService;
import com.llm_modifier.fearless_gpus.service.MinioService;
import com.llm_modifier.fearless_gpus.service.RabbitMQProducer;
import org.junit.jupiter.api.Test;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.*;

class DeletionContractTest {

    private static User student(long id) {
        return User.builder().id(id).email("student@example.com").role(Role.ROLE_STUDENT).build();
    }

    @Test
    void completedOwnedDatasetDeletesObjectAndRecord() {
        MinioService minio = mock(MinioService.class);
        DatasetRecordRepository datasets = mock(DatasetRecordRepository.class);
        RabbitMQProducer producer = mock(RabbitMQProducer.class);
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        DatasetRecord record = DatasetRecord.builder()
                .id(11L).user(owner).filename("train.jsonl")
                .minioObjectName("object-train.jsonl").status("COMPLETED").build();
        when(datasets.findById(11L)).thenReturn(Optional.of(record));
        when(jobs.existsByDatasetId(11L)).thenReturn(false);

        DatasetController controller = new DatasetController(minio, datasets, producer, jobs);
        assertEquals(204, controller.deleteDataset(11L, owner).getStatusCode().value());
        verify(minio).deleteFile("object-train.jsonl");
        verify(datasets).delete(record);
    }

    @Test
    void pendingOwnedDatasetCanBeDeleted() {
        MinioService minio = mock(MinioService.class);
        DatasetRecordRepository datasets = mock(DatasetRecordRepository.class);
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        DatasetRecord record = DatasetRecord.builder()
                .id(12L).user(owner).filename("HuggingFaceTB/smoltalk2")
                .source("HUGGINGFACE").minioObjectName("").status("PENDING").build();
        when(datasets.findById(12L)).thenReturn(Optional.of(record));
        when(jobs.existsByDatasetId(12L)).thenReturn(false);

        DatasetController controller = new DatasetController(minio, datasets, mock(RabbitMQProducer.class), jobs);
        assertEquals(204, controller.deleteDataset(12L, owner).getStatusCode().value());
        verify(minio).deleteFile("");
        verify(datasets).delete(record);
    }

    @Test
    void datasetReferencedByJobCannotBeDeleted() {
        MinioService minio = mock(MinioService.class);
        DatasetRecordRepository datasets = mock(DatasetRecordRepository.class);
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        DatasetRecord record = DatasetRecord.builder()
                .id(11L).user(owner).filename("train.jsonl").status("COMPLETED").build();
        when(datasets.findById(11L)).thenReturn(Optional.of(record));
        when(jobs.existsByDatasetId(11L)).thenReturn(true);
        DatasetController controller = new DatasetController(minio, datasets, mock(RabbitMQProducer.class), jobs);

        ResponseStatusException error = assertThrows(
                ResponseStatusException.class,
                () -> controller.deleteDataset(11L, owner)
        );
        assertEquals(409, error.getStatusCode().value());
        verify(datasets, never()).delete(any());
    }

    @Test
    void terminalOwnedJobCanBeDeleted() {
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        FinetuneJob job = FinetuneJob.builder()
                .jobUuid("job-1").user(owner).baseModelId("unsloth/model")
                .recipe("lora").status("COMPLETED").build();
        when(jobs.findByJobUuid("job-1")).thenReturn(Optional.of(job));
        FinetuneJobService service = new FinetuneJobService(
                jobs, mock(DatasetRecordRepository.class), mock(RabbitTemplate.class)
        );

        service.deleteJob("job-1", owner);
        verify(jobs).delete(job);
    }

    @Test
    void activeJobCannotBeDeleted() {
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        FinetuneJob job = FinetuneJob.builder()
                .jobUuid("job-1").user(owner).baseModelId("unsloth/model")
                .recipe("lora").status("RUNNING").updatedAt(LocalDateTime.now()).build();
        when(jobs.findByJobUuid("job-1")).thenReturn(Optional.of(job));
        FinetuneJobService service = new FinetuneJobService(
                jobs, mock(DatasetRecordRepository.class), mock(RabbitTemplate.class)
        );

        ResponseStatusException error = assertThrows(
                ResponseStatusException.class,
                () -> service.deleteJob("job-1", owner)
        );
        assertEquals(409, error.getStatusCode().value());
        verify(jobs, never()).delete(any());
    }

    @Test
    void abandonedQueuedJobCanBeDeleted() {
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        FinetuneJob job = FinetuneJob.builder()
                .jobUuid("job-queued").user(owner).baseModelId("unsloth/model")
                .recipe("lora").status("QUEUED").updatedAt(LocalDateTime.now()).build();
        when(jobs.findByJobUuid("job-queued")).thenReturn(Optional.of(job));
        FinetuneJobService service = new FinetuneJobService(
                jobs, mock(DatasetRecordRepository.class), mock(RabbitTemplate.class)
        );

        service.deleteJob("job-queued", owner);
        verify(jobs).delete(job);
    }

    @Test
    void staleInterruptedRunningJobCanBeDeleted() {
        FinetuneJobRepository jobs = mock(FinetuneJobRepository.class);
        User owner = student(7L);
        FinetuneJob job = FinetuneJob.builder()
                .jobUuid("job-stale").user(owner).baseModelId("unsloth/model")
                .recipe("lora").status("RUNNING").updatedAt(LocalDateTime.now().minusMinutes(2)).build();
        when(jobs.findByJobUuid("job-stale")).thenReturn(Optional.of(job));
        FinetuneJobService service = new FinetuneJobService(
                jobs, mock(DatasetRecordRepository.class), mock(RabbitTemplate.class)
        );

        service.deleteJob("job-stale", owner);
        verify(jobs).delete(job);
    }
}
