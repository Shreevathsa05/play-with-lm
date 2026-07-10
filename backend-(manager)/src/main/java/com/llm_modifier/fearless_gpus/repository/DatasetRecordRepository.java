package com.llm_modifier.fearless_gpus.repository;

import com.llm_modifier.fearless_gpus.model.DatasetRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface DatasetRecordRepository extends JpaRepository<DatasetRecord, Long> {
    List<DatasetRecord> findByUserId(Long userId);
}
