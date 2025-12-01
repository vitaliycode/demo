"""
Pydantic модели для валидации запросов и ответов.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HealthResponse(BaseModel):
    """Ответ проверки здоровья системы."""
    status: str
    model_loaded: bool
    device: str
    model_name: str
    version: str


class VQARequest(BaseModel):
    """Запрос на визуальные вопросы и ответы."""
    image: str = Field(..., description="Base64 закодированное изображение или URL")
    question: Optional[str] = Field("", description="Вопрос об изображении (пусто для создания описания)")
    session_id: Optional[str] = Field(None, description="ID сессии для кэширования изображения")
    max_tokens: Optional[int] = Field(None, description="Максимальное количество токенов для генерации")


class VQAResponse(BaseModel):
    """Ответ визуальных вопросов и ответов."""
    answer: str
    session_id: str
    timestamp: str
    processing_time: Optional[float] = None


class OCRRequest(BaseModel):
    """Запрос OCR (распознавание текста)."""
    image: str = Field(..., description="Base64 закодированное изображение или URL")
    language: Optional[str] = Field("en", description="Подсказка языка (en/ru)")


class OCRResponse(BaseModel):
    """Ответ OCR (распознавание текста)."""
    text: str
    download_url: str
    task_id: str
    processing_time: Optional[float] = None


class ImageCaptionRequest(BaseModel):
    """Запрос создания описания изображения."""
    image: str = Field(..., description="Base64 закодированное изображение или URL")
    session_id: Optional[str] = None


class ImageCaptionResponse(BaseModel):
    """Ответ создания описания изображения."""
    caption: str
    session_id: str
    timestamp: str
    processing_time: Optional[float] = None


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str
    message: str
    code: str
    timestamp: Optional[str] = None


class TaskSubmitRequest(BaseModel):
    """Общий запрос задачи через multipart form."""
    query: str = Field(..., description="Текстовый запрос/промпт")
    image: Optional[str] = Field(None, description="Данные изображения")


class TaskSubmitResponse(BaseModel):
    """Ответ отправки задачи."""
    result: str
    processing_time: Optional[float] = None
