"""
Утилиты для управления сессиями и результатами.
"""
import uuid
import time
from typing import Optional, Tuple, Dict
from PIL import Image
from datetime import datetime


# Хранилище в памяти (для демо - в production используйте Redis)
_sessions: Dict[str, Tuple[bytes, Image.Image, float]] = {}
_ocr_results: Dict[str, Tuple[str, float]] = {}


def generate_session_id() -> str:
    """Генерация уникального ID сессии."""
    return f"sess-{uuid.uuid4().hex[:16]}"


def generate_task_id() -> str:
    """Генерация уникального ID задачи."""
    return f"task-{uuid.uuid4().hex[:16]}"


def save_session_image(session_id: str, image_bytes: bytes, image: Image.Image) -> None:
    """
    Сохранение изображения в кэш сессии.
    
    Args:
        session_id: Идентификатор сессии
        image_bytes: Байты изображения
        image: Объект PIL Image
    """
    _sessions[session_id] = (image_bytes, image, time.time())


def get_session_image(session_id: str) -> Optional[Tuple[bytes, Image.Image]]:
    """
    Получение изображения из кэша сессии.
    
    Args:
        session_id: Идентификатор сессии
        
    Returns:
        Кортеж из (image_bytes, PIL.Image) или None если не найдено
    """
    if session_id in _sessions:
        image_bytes, image, _ = _sessions[session_id]
        return image_bytes, image
    return None


def save_ocr_result(text: str) -> str:
    """
    Сохранение результата OCR и возврат ID задачи.
    
    Args:
        text: Результат текста OCR
        
    Returns:
        ID задачи для загрузки результата
    """
    task_id = generate_task_id()
    _ocr_results[task_id] = (text, time.time())
    return task_id


def get_ocr_result(task_id: str) -> Optional[str]:
    """
    Получение результата OCR по ID задачи.
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        Текст OCR или None если не найден
    """
    if task_id in _ocr_results:
        text, _ = _ocr_results[task_id]
        return text
    return None


def cleanup_expired_sessions(timeout: int) -> None:
    """
    Удаление истекших сессий из кэша.
    
    Args:
        timeout: Таймаут сессии в секундах
    """
    current_time = time.time()
    expired = [
        sid for sid, (_, _, ts) in _sessions.items()
        if current_time - ts > timeout
    ]
    for sid in expired:
        del _sessions[sid]


def cleanup_expired_ocr_results(timeout: int) -> None:
    """
    Удаление истекших результатов OCR из кэша.
    
    Args:
        timeout: Таймаут результата в секундах
    """
    current_time = time.time()
    expired = [
        tid for tid, (_, ts) in _ocr_results.items()
        if current_time - ts > timeout
    ]
    for tid in expired:
        del _ocr_results[tid]


def format_timestamp() -> str:
    """Получение текущей временной метки в ISO формате."""
    return datetime.now().isoformat()
