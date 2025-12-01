"""
Управление конфигурацией для SmolVLM Demo.
Загружает настройки из переменных окружения с разумными значениями по умолчанию.
"""
import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из переменных окружения."""
    
    # Приложение
    APP_NAME: str = "SmolVLM Demo"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, validation_alias="DEBUG")
    
    # Сервер
    HOST: str = Field(default="0.0.0.0", validation_alias="HOST")
    PORT: int = Field(default=8000, validation_alias="PORT")
    WORKERS: int = Field(default=1, validation_alias="WORKERS")
    
    # Конфигурация модели
    MODEL_NAME: str = Field(
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        validation_alias="MODEL_NAME"
    )
    MODEL_SIZE: Literal["256M", "500M"] = Field(default="500M", validation_alias="MODEL_SIZE")
    DEVICE: Literal["cuda", "cpu"] = Field(default="cpu", validation_alias="DEVICE")
    TORCH_DTYPE: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        validation_alias="TORCH_DTYPE"
    )
    
    # Кэш модели
    HF_HOME: str = Field(default="/data/hf-cache", validation_alias="HF_HOME")
    TRANSFORMERS_CACHE: str = Field(
        default="/data/hf-cache",
        validation_alias="TRANSFORMERS_CACHE"
    )
    HF_LOCAL_ONLY: bool = Field(default=False, validation_alias="HF_LOCAL_ONLY")
    
    # Настройки инференса
    MAX_NEW_TOKENS: int = Field(default=512, validation_alias="MAX_NEW_TOKENS")
    TEMPERATURE: float = Field(default=0.7, validation_alias="TEMPERATURE")
    REPETITION_PENALTY: float = Field(default=1.2, validation_alias="REPETITION_PENALTY")
    INFERENCE_TIMEOUT: int = Field(default=300, validation_alias="INFERENCE_TIMEOUT")
    
    # Лимиты изображений
    MAX_IMAGE_SIZE: int = Field(default=10 * 1024 * 1024, validation_alias="MAX_IMAGE_SIZE")  # 10MB
    MAX_IMAGE_DIMENSION: int = Field(default=4096, validation_alias="MAX_IMAGE_DIMENSION")
    ALLOWED_IMAGE_TYPES: set = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    
    # Управление сессиями
    SESSION_TIMEOUT: int = Field(default=3600, validation_alias="SESSION_TIMEOUT")  # 1 час
    
    # CORS
    CORS_ORIGINS: list = Field(
        default=["*"],
        validation_alias="CORS_ORIGINS"
    )
    
    # Конфигурация UI
    ENABLE_GRADIO: bool = Field(default=True, validation_alias="ENABLE_GRADIO")
    GRADIO_PORT: int = Field(default=7860, validation_alias="GRADIO_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Глобальный экземпляр настроек
settings = Settings()


def get_model_id_by_size(size: str) -> str:
    """Получить полный ID модели по сокращенному размеру."""
    model_map = {
        "256M": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    }
    return model_map.get(size, settings.MODEL_NAME)

