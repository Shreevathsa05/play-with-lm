package com.llm_modifier.fearless_gpus.repository;

import com.llm_modifier.fearless_gpus.model.FinetuneJob;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface FinetuneJobRepository extends JpaRepository<FinetuneJob, Long> {
    Optional<FinetuneJob> findByJobUuid(String jobUuid);
    List<FinetuneJob> findByUserIdOrderByCreatedAtDesc(Long userId);
    boolean existsByDatasetId(Long datasetId);
}
