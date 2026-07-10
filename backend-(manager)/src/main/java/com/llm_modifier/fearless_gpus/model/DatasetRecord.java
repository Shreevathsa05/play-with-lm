package com.llm_modifier.fearless_gpus.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "datasets")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DatasetRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String filename;

    @Column(nullable = true)
    private String minioObjectName;

    @Column(nullable = true)
    private String source = "MINIO"; // "MINIO" or "HUGGINGFACE"

    @Column(nullable = true)
    private String huggingFaceId;

    @Column(nullable = false)
    private String status;

    @Column(columnDefinition = "TEXT")
    private String auditReportJson;

    private LocalDateTime uploadedAt;

    @PrePersist
    public void prePersist() {
        this.uploadedAt = LocalDateTime.now();
    }
}
