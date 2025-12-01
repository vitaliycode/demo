"""
Утилиты валидации входных данных.
"""
import base64
import io
import requests
from PIL import Image
from typing import Tuple
from urllib.parse import urlparse
from app.config import settings


class ValidationError(Exception):
    """Пользовательская ошибка валидации."""
    def __init__(self, error: str, message: str, code: str):
        self.error = error
        self.message = message
        self.code = code
        super().__init__(message)


def validate_base64_image(image_data: str) -> Tuple[bytes, Image.Image]:
    """
    Валидация и декодирование base64 изображения или загрузка по URL.
    
    Args:
        image_data: Base64 закодированное изображение или URL
        
    Returns:
        Кортеж из (image_bytes, PIL.Image)
        
    Raises:
        ValidationError: Если изображение невалидно
    """
    # Проверка на URL
    if image_data.startswith(('http://', 'https://')):
        try:
            response = requests.get(
                image_data,
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            image_bytes = response.content
        except Exception as e:
            raise ValidationError(
                error="INVALID_URL",
                message=f"Не удалось загрузить изображение по URL: {str(e)}",
                code="URL_ERROR"
            )
    else:
        # Обработка base64 данных (с префиксом или без)
        try:
            # Удаление префикса data URL если присутствует
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception as e:
            raise ValidationError(
                error="INVALID_BASE64",
                message=f"Неверное base64 кодирование: {str(e)}",
                code="DECODE_ERROR"
            )
    
    # Проверка размера
    if len(image_bytes) > settings.MAX_IMAGE_SIZE:
        raise ValidationError(
            error="IMAGE_TOO_LARGE",
            message=f"Размер изображения {len(image_bytes)} превышает максимум {settings.MAX_IMAGE_SIZE}",
            code="SIZE_ERROR"
        )
    
    # Попытка открыть изображение
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Проверка на валидность изображения
        image = Image.open(io.BytesIO(image_bytes))  # Перезагрузка после verify
    except Exception as e:
        raise ValidationError(
            error="INVALID_IMAGE",
            message=f"Неверный формат изображения: {str(e)}",
            code="FORMAT_ERROR"
        )
    
    # Проверка размеров
    if max(image.size) > settings.MAX_IMAGE_DIMENSION:
        raise ValidationError(
            error="IMAGE_TOO_LARGE",
            message=f"Размер изображения {max(image.size)} превышает максимум {settings.MAX_IMAGE_DIMENSION}",
            code="DIMENSION_ERROR"
        )
    
    # Конвертация в RGB если необходимо
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    return image_bytes, image


def validate_language(lang: str) -> str:
    """
    Валидация кода языка.
    
    Args:
        lang: Код языка
        
    Returns:
        Валидированный код языка
    """
    supported_languages = {'en', 'ru', 'auto'}
    lang = lang.lower().strip()
    
    if lang not in supported_languages:
        # По умолчанию auto если не поддерживается
        return 'auto'
    
    return lang


def validate_session_id(session_id: str) -> str:
    """
    Валидация формата ID сессии.
    
    Args:
        session_id: Идентификатор сессии
        
    Returns:
        Валидированный ID сессии
        
    Raises:
        ValidationError: Если ID сессии невалиден
    """
    if not session_id or len(session_id) > 128:
        raise ValidationError(
            error="INVALID_SESSION",
            message="ID сессии должен быть от 1 до 128 символов",
            code="SESSION_ERROR"
        )
    
    # Разрешены только буквы, цифры и дефисы
    if not all(c.isalnum() or c in '-_' for c in session_id):
        raise ValidationError(
            error="INVALID_SESSION",
            message="ID сессии должен содержать только буквы, цифры, дефисы и подчеркивания",
            code="SESSION_ERROR"
        )
    
    return session_id
