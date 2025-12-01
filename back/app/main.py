"""
FastAPI приложение для SmolVLM Demo.
Предоставляет REST API endpoints и обслуживает frontend.
"""
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from gradio.routes import mount_gradio_app
import os

from app.config import settings
from app.schemas import (
    VQARequest, VQAResponse,
    OCRRequest, OCRResponse,
    ImageCaptionRequest, ImageCaptionResponse,
    HealthResponse, ErrorResponse,
    TaskSubmitResponse
)
from app.validators import validate_base64_image, validate_language, ValidationError
from app.utils import (
    generate_session_id, save_session_image, get_session_image,
    save_ocr_result, get_ocr_result,
    cleanup_expired_sessions, cleanup_expired_ocr_results,
    format_timestamp
)
from app.model_manager import ModelManager


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Экземпляр менеджера модели
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Менеджер жизненного цикла приложения."""
    # Запуск
    logger.info("Starting SmolVLM Demo application...")
    asyncio.create_task(load_model_async())
    asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Завершение
    logger.info("Shutting down SmolVLM Demo application...")


# Создание FastAPI приложения
app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal vision-language model demo with VQA, OCR, and captioning capabilities",
    version=settings.VERSION,
    lifespan=lifespan
)


# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Подключение статических файлов
static_dir = "/app/front"
if not os.path.exists(static_dir):
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "front")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


async def load_model_async():
    """Асинхронная загрузка модели при запуске."""
    try:
        logger.info("Loading model asynchronously...")
        await asyncio.to_thread(model_manager.load_model)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


async def periodic_cleanup():
    """Периодическая очистка истекших сессий и результатов."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            cleanup_expired_sessions(settings.SESSION_TIMEOUT)
            cleanup_expired_ocr_results(settings.SESSION_TIMEOUT)
            logger.debug("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Отдача HTML страницы frontend."""
    if os.path.exists(static_dir):
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return f.read()
    
    return """
    <html>
        <head><title>SmolVLM Demo</title></head>
        <body>
            <h1>SmolVLM Demo</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
            <p>Visit <a href="/ui">/ui</a> for Gradio interface.</p>
        </body>
    </html>
    """


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns application status and model information.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "loading",
        model_loaded=model_manager.is_loaded(),
        device=settings.DEVICE,
        model_name=settings.MODEL_NAME,
        version=settings.VERSION
    )


@app.post("/api/vqa", response_model=VQAResponse)
async def visual_question_answering(request: VQARequest):
    """
    Visual Question Answering endpoint.
    
    Accepts an image and a question, returns the model's answer.
    Supports session caching for multiple questions on the same image.
    """
    start_time = time.time()
    
    try:
        # Валидация и обработка изображения
        image_bytes, image = validate_base64_image(request.image)
        
        # Обработка сессии
        session_id = request.session_id
        if session_id:
            existing = get_session_image(session_id)
            if existing:
                image_bytes, image = existing
            else:
                save_session_image(session_id, image_bytes, image)
        else:
            session_id = generate_session_id()
            save_session_image(session_id, image_bytes, image)
        
        # Запуск инференса
        try:
            answer = await asyncio.to_thread(
                model_manager.vqa_inference,
                image,
                request.question,
                request.max_tokens
            )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="INFERENCE_ERROR",
                    message=f"Model inference failed: {str(e)}",
                    code="MODEL_ERROR",
                    timestamp=format_timestamp()
                ).dict()
            )
        
        processing_time = time.time() - start_time
        
        return VQAResponse(
            answer=answer,
            session_id=session_id,
            timestamp=format_timestamp(),
            processing_time=processing_time
        )
    
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=e.error,
                message=e.message,
                code=e.code,
                timestamp=format_timestamp()
            ).dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message=f"Internal server error: {str(e)}",
                code="SERVER_ERROR",
                timestamp=format_timestamp()
            ).dict()
        )


@app.post("/api/ocr", response_model=OCRResponse)
async def optical_character_recognition(request: OCRRequest):
    """
    OCR (Optical Character Recognition) endpoint.
    
    Extracts text from the provided image.
    """
    start_time = time.time()
    
    try:
        # Валидация изображения
        image_bytes, image = validate_base64_image(request.image)
        language = validate_language(request.language)
        
        # Запуск OCR
        try:
            text = await asyncio.to_thread(
                model_manager.ocr_inference,
                image
            )
        except Exception as e:
            logger.error(f"OCR error: {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="OCR_ERROR",
                    message=f"OCR processing failed: {str(e)}",
                    code="OCR_ERROR",
                    timestamp=format_timestamp()
                ).dict()
            )
        
        # Сохранение результата для скачивания
        task_id = save_ocr_result(text)
        download_url = f"/api/download/ocr/{task_id}"
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            text=text,
            download_url=download_url,
            task_id=task_id,
            processing_time=processing_time
        )
    
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=e.error,
                message=e.message,
                code=e.code,
                timestamp=format_timestamp()
            ).dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message=f"Internal server error: {str(e)}",
                code="SERVER_ERROR",
                timestamp=format_timestamp()
            ).dict()
        )


@app.post("/api/caption", response_model=ImageCaptionResponse)
async def image_captioning(request: ImageCaptionRequest):
    """
    Image Captioning endpoint.
    
    Generates a descriptive caption for the provided image.
    """
    start_time = time.time()
    
    try:
        # Валидация изображения
        image_bytes, image = validate_base64_image(request.image)
        
        # Обработка сессии
        session_id = request.session_id or generate_session_id()
        save_session_image(session_id, image_bytes, image)
        
        # Генерация подписи
        try:
            caption = await asyncio.to_thread(
                model_manager.caption_inference,
                image
            )
        except Exception as e:
            logger.error(f"Caption error: {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="CAPTION_ERROR",
                    message=f"Caption generation failed: {str(e)}",
                    code="CAPTION_ERROR",
                    timestamp=format_timestamp()
                ).dict()
            )
        
        processing_time = time.time() - start_time
        
        return ImageCaptionResponse(
            caption=caption,
            session_id=session_id,
            timestamp=format_timestamp(),
            processing_time=processing_time
        )
    
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=e.error,
                message=e.message,
                code=e.code,
                timestamp=format_timestamp()
            ).dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message=f"Internal server error: {str(e)}",
                code="SERVER_ERROR",
                timestamp=format_timestamp()
            ).dict()
        )


@app.get("/api/download/ocr/{task_id}")
async def download_ocr_result(task_id: str):
    """
    Download OCR result as text file.
    
    Args:
        task_id: Task identifier from OCR response
    """
    text = get_ocr_result(task_id)
    
    if text is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="NOT_FOUND",
                message="OCR result not found or expired",
                code="TASK_NOT_FOUND",
                timestamp=format_timestamp()
            ).dict()
        )
    
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="ocr_result_{task_id}.txt"'
        }
    )


@app.post("/ptt/convert", response_model=TaskSubmitResponse)
async def ptt_convert(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Legacy endpoint for compatibility with SmolVLM2-Demo.
    
    Accepts multipart form data with query and optional image.
    """
    start_time = time.time()
    
    try:
        if image:
            # Чтение файла изображения
            image_bytes = await image.read()
            
            # Конвертация в base64 для валидации
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Валидация
            _, img = validate_base64_image(image_b64)
            
            # Запуск VQA
            result = await asyncio.to_thread(
                model_manager.vqa_inference,
                img,
                query
            )
        else:
            # Возврат сообщения если нет изображения
            result = "Please provide an image for analysis."
        
        processing_time = time.time() - start_time
        
        return TaskSubmitResponse(
            result=result,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"PTT convert error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =============================================================================
# Gradio UI Integration
# =============================================================================

if settings.ENABLE_GRADIO:
    try:
        from app.gradio_ui import create_gradio_interface
        
        logger.info("Mounting Gradio UI...")
        gradio_app = create_gradio_interface(model_manager)
        mount_gradio_app(app, gradio_app, path="/test")
        logger.info("Gradio UI mounted at /test")
    except ImportError as e:
        logger.warning(f"Gradio UI not available: {e}")
    except Exception as e:
        logger.error(f"Failed to mount Gradio UI: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG
    )

