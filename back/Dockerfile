# CPU-оптимизированный Dockerfile для SmolVLM Demo

FROM python:3.10-slim

# Установка рабочей директории
WORKDIR /app

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    DEVICE=cpu \
    TORCH_DTYPE=float32

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY back/requirements.txt back/requirements-cpu.txt ./

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY back/app/ ./app/

# Копирование frontend
COPY front/ /app/front/

# Создание директорий для результатов
RUN mkdir -p /app/outputs/chat /app/outputs/ocr

# Открытие порта
EXPOSE 8000

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Запуск приложения с логированием
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

