"""
Менеджер модели для загрузки и выполнения инференса с моделями SmolVLM.
Реализует паттерн singleton с поддержкой нескольких размеров моделей.
"""
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Optional
import logging
from PIL import Image

from app.config import settings, get_model_id_by_size


logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton менеджер модели для SmolVLM."""
    
    _instance: Optional['ModelManager'] = None
    _model: Optional[AutoModelForVision2Seq] = None
    _processor: Optional[AutoProcessor] = None
    _loaded: bool = False
    _current_model_id: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация менеджера модели."""
        pass
    
    def load_model(self, model_id: Optional[str] = None) -> None:
        """
        Загрузка модели и процессора из кэша или HuggingFace Hub.
        
        Args:
            model_id: Конкретный ID модели для загрузки, или None для использования конфига
        """
        if model_id is None:
            model_id = get_model_id_by_size(settings.MODEL_SIZE)
        
        # Проверка на уже загруженную модель
        if self._model is not None and self._current_model_id == model_id:
            logger.info(f"Модель {model_id} уже загружена")
            return
        
        logger.info(f"Загрузка модели {model_id}...")
        
        # Определение устройства и типа данных
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            device = "cuda"
            if settings.TORCH_DTYPE == "float16":
                torch_dtype = torch.float16
            elif settings.TORCH_DTYPE == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32
            if settings.DEVICE == "cuda":
                logger.warning("Запрошена CUDA, но не доступна. Используется CPU.")
        
        logger.info(f"Устройство: {device}, тип данных: {torch_dtype}")
        
        # Загрузка процессора
        logger.info("Загрузка процессора...")
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=settings.TRANSFORMERS_CACHE,
            local_files_only=settings.HF_LOCAL_ONLY,
            trust_remote_code=True
        )
        
        # Загрузка модели
        logger.info("Загрузка модели...")
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            cache_dir=settings.TRANSFORMERS_CACHE,
            local_files_only=settings.HF_LOCAL_ONLY,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Перенос на устройство
        self._model = self._model.to(device)
        self._model.eval()
        
        self._current_model_id = model_id
        self._loaded = True
        
        logger.info(f"Модель успешно загружена на {device}")
    
    def is_loaded(self) -> bool:
        """Проверка загрузки модели."""
        return self._loaded and self._model is not None
    
    def get_model(self) -> AutoModelForVision2Seq:
        """Получение загруженной модели."""
        if not self.is_loaded():
            self.load_model()
        return self._model
    
    def get_processor(self) -> AutoProcessor:
        """Получение загруженного процессора."""
        if not self.is_loaded():
            self.load_model()
        return self._processor
    
    def vqa_inference(
        self,
        image: Image.Image,
        question: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Выполнение инференса Visual Question Answering.
        
        Args:
            image: PIL изображение
            question: Вопрос об изображении
            max_tokens: Максимум токенов для генерации
            
        Returns:
            Ответ модели
        """
        if not self.is_loaded():
            self.load_model()
        
        model = self.get_model()
        processor = self.get_processor()
        
        if not question.strip():
            question = "Опиши это изображение подробно."
        
        # Подготовка сообщений
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Применение шаблона чата
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Обработка входных данных
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        # Перенос на устройство
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Генерация
        max_new_tokens = max_tokens or settings.MAX_NEW_TOKENS
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=settings.TEMPERATURE,
                repetition_penalty=settings.REPETITION_PENALTY,
            )
        
        # Декодирование
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Извлечение ответа
        full_response = generated_texts[0]
        
        # Удаление промпта из ответа
        if prompt in full_response:
            answer = full_response.split(prompt, 1)[-1].strip()
        elif question in full_response:
            answer = full_response.split(question, 1)[-1].strip()
        else:
            answer = full_response
        
        # Очистка ответа
        answer = self._clean_response(answer)
        
        return answer
    
    def ocr_inference(self, image: Image.Image) -> str:
        """
        Извлечение текста из изображения с помощью OCR.
        
        Args:
            image: PIL изображение
            
        Returns:
            Извлеченный текст
        """
        if not self.is_loaded():
            self.load_model()
        
        model = self.get_model()
        processor = self.get_processor()
        
        question = "Извлеки весь текст с этого изображения. Верни только текст, без дополнительного описания."
        
        # Подготовка сообщений
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Применение шаблона чата
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Обработка входных данных
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        # Перемещение на устройство
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Генерация
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                temperature=settings.TEMPERATURE,
                repetition_penalty=settings.REPETITION_PENALTY,
            )
        
        # Декодирование
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Извлечение текста
        full_response = generated_texts[0]
        
        # Удаление промпта
        if prompt in full_response:
            text = full_response.split(prompt, 1)[-1].strip()
        elif question in full_response:
            text = full_response.split(question, 1)[-1].strip()
        else:
            text = full_response
        
        # Очистка
        text = self._clean_response(text)
        
        # Дополнительная очистка для OCR
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def caption_inference(self, image: Image.Image) -> str:
        """
        Генерация описания изображения.
        
        Args:
            image: PIL изображение
            
        Returns:
            Описание изображения
        """
        return self.vqa_inference(image, "Опиши это изображение подробно.")
    
    def _clean_response(self, text: str) -> str:
        """
        Очистка ответа модели путем удаления специальных токенов и артефактов.
        
        Args:
            text: Необработанный вывод модели
            
        Returns:
            Очищенный текст
        """
        # Удаление распространенных артефактов
        artifacts = [
            "Assistant:", "assistant:",
            "<text>", "</text>",
            "<|im_start|>", "<|im_end|>",
            "<end_of_utterance>",
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        text = text.strip()
        
        # Удаление повторяющихся окончаний
        words = text.split()
        if len(words) > 3:
            last_words = words[-3:]
            if len(set(last_words)) == 1 and len(words) > 10:
                # Поиск начала повторения
                for i in range(len(words) - 3, 0, -1):
                    if words[i:i+3] == last_words:
                        text = " ".join(words[:i])
                        break
        
        # Нормализация пробелов
        text = " ".join(text.split())
        
        return text

