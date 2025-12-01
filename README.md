# SmolVLM Demo

## Быстрый старт

### CPU режим (рекомендуется для начала)

```bash
./start.sh
```

### GPU режим (требуется NVIDIA GPU)

```bash
./start.sh gpu
```

После запуска откройте:
- **Веб-интерфейс**: http://localhost:8000
- **Gradio UI**: http://localhost:8000/gradio
- **API документация**: http://localhost:8000/docs

## Возможности

### VQA (Visual Question Answering)
Задавайте вопросы об изображениях и получайте интеллектуальные ответы от модели.

### OCR (Распознавание текста)
Извлекайте текст из изображений с высокой точностью.

### Описание изображений
Автоматическая генерация описаний для загруженных изображений.


## Требования

### Минимальные требования
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM (для CPU режима)
- 10GB свободного дискового пространства

### Для GPU режима
- NVIDIA GPU с 6GB+ VRAM
- NVIDIA Container Toolkit
- CUDA 11.8+

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd demo
```

2. (Опционально) Создайте файл конфигурации:
```bash
cp env.example .env
```

3. Запустите проект:
```bash
./start.sh      # CPU режим
./start.sh gpu  # GPU режим
```

При первом запуске автоматически:
- Соберётся Docker образ
- Скачается модель SmolVLM2 (~1-2GB)
- Запустится веб-сервер

## Использование

### Веб-интерфейс (http://localhost:8000)

### Gradio UI (http://localhost:8000/gradio)

### REST API (http://localhost:8000/docs)

## Управление

### Основные команды

```bash
# Запуск
./start.sh          # CPU
./start.sh gpu      # GPU

# Остановка
./stop.sh

# Сборка (опционально, start.sh собирает автоматически)
./build.sh          # CPU
./build.sh gpu      # GPU

# Просмотр логов
docker logs -f smolvlm-demo-cpu     # CPU
docker logs -f smolvlm-demo-gpu     # GPU
```

### Проверка статуса

```bash
# Статус контейнеров
docker ps

# Проверка сервиса
curl http://localhost:8000/api/health
```

## Конфигурация

### Переменные окружения (.env)

```bash
# Размер модели (256M или 500M)
MODEL_SIZE=500M

# Устройство (cpu или cuda)
DEVICE=cpu

# Максимальное количество токенов в ответе
MAX_NEW_TOKENS=512

# Температура генерации (0.0-1.0)
TEMPERATURE=0.7

# Порт веб-сервера
PORT=8000

# Offline режим (1 - не скачивать модель из интернета)
HF_LOCAL_ONLY=0
```
**Версия**: 1.0.1  
