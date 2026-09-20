# Multi-stage build for Spring Boot manager
FROM maven:3.9.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml .
COPY .mvn .mvn
COPY mvnw mvnw
COPY src ./src
RUN chmod +x mvnw && ./mvnw -q -DskipTests package

FROM eclipse-temurin:21-jre-jammy
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 appuser
COPY --from=build /app/target/fearless-gpus-*.jar /app/app.jar
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --retries=12 --start-period=90s \
  CMD curl -fsS -o /dev/null -w "%{http_code}" -X POST http://localhost:8080/api/auth/login \
      -H "Content-Type: application/json" -d '{}' | grep -Eq "400|401|403" || exit 1
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
