# The runtime image must use an explicitly selected, tested hardware stack.
# Override REQUIREMENTS_FILE for a certified CPU/dry-run image.
FROM python:3.12-slim

ARG REQUIREMENTS_FILE=requirements-nvidia.txt

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CUSTOMIZER_DRY_RUN=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}
RUN if [ "${REQUIREMENTS_FILE}" = "requirements-nvidia.txt" ] || [ "${REQUIREMENTS_FILE}" = "requirements-amd.txt" ]; then \
      pip check && \
      python -c "import torch, torchao, peft; from transformers.quantizers.auto import AutoHfQuantizer; from peft.import_utils import is_torchao_available; assert is_torchao_available(), 'PEFT rejected the installed TorchAO version'; print(f'Validated ML stack: torch={torch.__version__}, torchao={torchao.__version__}, peft={peft.__version__}')"; \
    fi

COPY src ./src

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=20s \
  CMD python -c "import src.consumer, src.core.config; print('ok')" || exit 1

CMD ["python", "-m", "src.main"]
