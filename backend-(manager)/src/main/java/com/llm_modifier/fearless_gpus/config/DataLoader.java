package com.llm_modifier.fearless_gpus.config;

import com.llm_modifier.fearless_gpus.model.Role;
import com.llm_modifier.fearless_gpus.model.User;
import com.llm_modifier.fearless_gpus.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class DataLoader implements CommandLineRunner {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) throws Exception {
        if (userRepository.count() == 0) {
            User admin = User.builder()
                    .email("admin@example.com")
                    .password(passwordEncoder.encode("admin123"))
                    .role(Role.ROLE_ADMIN)
                    .build();
            userRepository.save(admin);

            User student = User.builder()
                    .email("student@example.com")
                    .password(passwordEncoder.encode("student123"))
                    .role(Role.ROLE_STUDENT)
                    .build();
            userRepository.save(student);
            
            System.out.println("Default users created: admin@example.com / admin123  &  student@example.com / student123");
        }
    }
}
