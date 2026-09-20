package com.llm_modifier.fearless_gpus.model;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "finetune_jobs")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FinetuneJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 64)
    private String jobUuid;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "dataset_id")
    private DatasetRecord dataset;

    @Column(nullable = false)
    private String baseModelId;

    @Column(nullable = false)
    private String recipe;

    @Column(nullable = false)
    private String objective;

    private String rewardModelId;

    @Column(columnDefinition = "TEXT")
    private String resumeFromCheckpoint;

    @Column(name = "model_params_b", nullable = true)
    private String modelParamsB;

    @Column(nullable = true)
    private String datasetMinioUri;

    @Column(nullable = true)
    private String exportHfRepo;

    @Column(nullable = true)
    private String exportFormat;

    @Column(columnDefinition = "TEXT")
    private String evalPromptsJson;

    @Column(nullable = false)
    private String status;

    @Column(columnDefinition = "TEXT")
    private String reportJson;

    @Column(columnDefinition = "TEXT")
    private String errorMessage;

    @Column(nullable = true, length = 64)
    private String attemptId;

    @Column(nullable = false)
    private long lastEventSeq = 0;

    private LocalDateTime leaseExpiresAt;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @PrePersist
    public void prePersist() {
        LocalDateTime now = LocalDateTime.now();
        this.createdAt = now;
        this.updatedAt = now;
        if (this.status == null) {
            this.status = "PENDING";
        }
    }

    @PreUpdate
    public void preUpdate() {
        this.updatedAt = LocalDateTime.now();
    }
}
