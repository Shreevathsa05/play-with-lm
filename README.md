# LLM Customizer - GPU Efficiency Orchestrator

This project is a platform that lets students upload datasets for standardization and optimization, pushing processed outputs to MinIO. Admins can audit all uploads, users, and queues.

## Default Credentials

For testing login/logout flows, the system automatically bootstraps the following accounts on startup:

- **Admin Account**:
  - Email: `admin@example.com`
  - Password: `admin123`
- **Student Account**:
  - Email: `student@example.com`
  - Password: `student123`

---

## How to Run the Application

Follow these steps to spin up the local environment and run both the frontend and backend.

### 1. Spin up the Local Infrastructure
The project root contains a `docker-compose.yml` configured with PostgreSQL, MinIO, and RabbitMQ.

Run the following command in the project root directory:
```bash
docker compose up
```

### 2. Run the Spring Boot Backend (Manager)
Navigate to the `backend-(manager)` directory and run the Spring Boot application:
```bash
cd "backend-(manager)"
chmod +x mvnw  # Ensure wrapper is executable
./mvnw spring-boot:run
```
The backend runs on `http://localhost:8080`.

### 3. Run the React Frontend (Interactor)
Navigate to the `ui-(interactor)` directory, install dependencies, and start the development server:
```bash
cd "ui-(interactor)"
npm install
npm run dev
```
The frontend will start on `http://localhost:5173`. Open this URL in your browser to log in.

### 4. Run the Python Data Processor
Navigate to the `data-processor` directory, activate the virtual environment, install dependencies, and start the processing service:
```bash
cd data-processor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.main
```
The Python service will connect to RabbitMQ and start listening for dataset jobs on `http://0.0.0.0:8000`.

---

## System Architecture

- **`backend-(manager)`**: Spring Boot service handling authentication, RBAC, metadata storage (PostgreSQL), object storage orchestration (MinIO), and task queuing.
- **`ui-(interactor)`**: Vite-based React application with authentication routing, student upload dashboard, and admin metrics view.
- **`data-processor`**: Python-based microservice that consumes normalization/standardization jobs from RabbitMQ, standardizes datasets (Alpaca/ShareGPT format), fetches HuggingFace datasets, and calculates dataset health and optimization metrics.
