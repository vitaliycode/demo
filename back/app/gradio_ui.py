"""
Gradio UI для SmolVLM Demo.
Предоставляет интерактивный веб-интерфейс для VQA и OCR.
"""
import gradio as gr
from PIL import Image
from typing import Optional, List, Tuple
import time
from pathlib import Path

from app.config import settings
from app.model_manager import ModelManager


def create_gradio_interface(model_manager: ModelManager):
    """
    Создание Gradio интерфейса для SmolVLM Demo.
    
    Args:
        model_manager: Экземпляр ModelManager
        
    Returns:
        Gradio Blocks интерфейс
    """
    
    # Хранилище истории чата для каждой сессии
    _chat_sessions = {}
    
    def chat_vqa(
        image: Optional[Image.Image],
        message: str,
        history: Optional[List] = None
    ) -> Tuple[List, str, Optional[str]]:
        """
        Обработка VQA чат взаимодействия.
        
        Args:
            image: PIL изображение
            message: Сообщение пользователя
            history: История чата в формате [(user_msg, bot_msg), ...]
            
        Returns:
            Кортеж из (обновленная_история, очищенный_ввод, файл_для_скачивания)
        """
        if history is None:
            history = []
        
        message = (message or "").strip()
        
        if not message:
            return history, "", None
        
        if not image:
            # Gradio Chatbot ожидает список кортежей (user_msg, bot_msg)
            history.append((message, "Пожалуйста, сначала загрузите изображение."))
            return history, "", None
        
        try:
            # Запуск инференса
            start_time = time.time()
            answer = model_manager.vqa_inference(image, message)
            processing_time = time.time() - start_time
            
            # Добавление в историю в формате (user_message, bot_response)
            history.append((message, answer))
            
            # Сохранение в файл
            timestamp = int(time.time())
            out_dir = Path("outputs/chat")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"chat_result_{timestamp}.txt"
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Question: {message}\n\n")
                f.write(f"Answer: {answer}\n\n")
                f.write(f"Processing time: {processing_time:.2f}s\n")
            
            return history, "", str(out_path)
        
        except Exception as e:
            # В случае ошибки также используем формат кортежа
            history.append((message, f"Ошибка: {str(e)}"))
            return history, "", None
    
    def run_ocr(image: Optional[Image.Image]) -> Tuple[str, Optional[str]]:
        """
        Запуск OCR на изображении.
        
        Args:
            image: PIL изображение
            
        Returns:
            Кортеж из (извлеченный_текст, файл_для_скачивания)
        """
        if not image:
            return "Пожалуйста, загрузите изображение с текстом.", None
        
        try:
            # Запуск OCR
            start_time = time.time()
            text = model_manager.ocr_inference(image)
            processing_time = time.time() - start_time
            
            # Сохранение в файл
            timestamp = int(time.time())
            out_dir = Path("outputs/ocr")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"ocr_result_{timestamp}.txt"
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.write(f"\n\n--- Processing time: {processing_time:.2f}s ---\n")
            
            return text, str(out_path)
        
        except Exception as e:
            return f"Ошибка: {str(e)}", None
    
    def generate_caption(image: Optional[Image.Image]) -> str:
        """
        Генерация подписи к изображению.
        
        Args:
            image: PIL изображение
            
        Returns:
            Текст подписи
        """
        if not image:
            return "Пожалуйста, загрузите изображение."
        
        try:
            caption = model_manager.caption_inference(image)
            return caption
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    # Пользовательский CSS
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto;
    }
    
    .header-text {
        text-align: center;
        padding: 20px;
    }
    
    .tab-content {
        padding: 15px;
    }
    
    .card {
        border-radius: 10px;
        border: 1px solid var(--border-color-primary);
        padding: 15px;
        background: var(--background-fill-secondary);
    }
    
    .info-box {
        background: var(--background-fill-secondary);
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    """
    
    # Создание Gradio интерфейса
    with gr.Blocks(title=f"{settings.APP_NAME}") as demo:
        # Применяем CSS после создания
        demo.css = custom_css
        
        # Заголовок
        gr.Markdown(
            f"""
            <div class="header-text">
            
            #{settings.APP_NAME}
            
            **Model:** `{settings.MODEL_NAME}`  
            **Device:** `{settings.DEVICE}` | **Version:** `{settings.VERSION}`
            
            Multimodal Vision-Language Model for VQA, OCR, and Image Captioning
            
            </div>
            """
        )
        
        # Вкладка VQA
        with gr.Tab("VQA"):
            gr.Markdown(
                """
                <div class="info-box">
                Загрузите изображение и задавайте вопросы о нём. Модель проанализирует изображение и предоставит детальные ответы.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    vqa_image = gr.Image(
                        label="Загрузите изображение",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    vqa_chatbot = gr.Chatbot(
                        label="Диалог",
                        height=500
                    )
            
            with gr.Row():
                vqa_input = gr.Textbox(
                    label="Ваш вопрос",
                    placeholder="Задайте любой вопрос об изображении...",
                    lines=2
                )
            
            with gr.Row():
                vqa_submit = gr.Button("Отправить", variant="primary", scale=2)
                vqa_clear = gr.Button("Очистить чат", scale=1)
            
            with gr.Row():
                vqa_file = gr.File(label="Скачать последний ответ")
            
            # Обработчики VQA
            vqa_submit.click(
                fn=chat_vqa,
                inputs=[vqa_image, vqa_input, vqa_chatbot],
                outputs=[vqa_chatbot, vqa_input, vqa_file]
            )
            
            vqa_input.submit(
                fn=chat_vqa,
                inputs=[vqa_image, vqa_input, vqa_chatbot],
                outputs=[vqa_chatbot, vqa_input, vqa_file]
            )
            
            vqa_clear.click(
                lambda: ([], None),
                outputs=[vqa_chatbot, vqa_file]
            )
        
        # Вкладка OCR
        with gr.Tab("OCR"):
            gr.Markdown(
                """
                <div class="info-box">
                Извлечение текста из изображений. Загрузите изображение с текстом и получите распознанный текст.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ocr_image = gr.Image(
                        label="Загрузите изображение с текстом",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    ocr_output = gr.Textbox(
                        label="Извлечённый текст",
                        lines=20,
                        max_lines=30
                    )
            
            with gr.Row():
                ocr_button = gr.Button("Извлечь текст", variant="primary")
            
            with gr.Row():
                ocr_file = gr.File(label="Скачать результат")
            
            # Обработчик OCR
            ocr_button.click(
                fn=run_ocr,
                inputs=[ocr_image],
                outputs=[ocr_output, ocr_file]
            )
        
        # Вкладка Captioning
        with gr.Tab("Описание"):
            gr.Markdown(
                """
                <div class="info-box">
                Автоматическая генерация описания изображений.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    caption_image = gr.Image(
                        label="Загрузите изображение",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    caption_output = gr.Textbox(
                        label="Сгенерированное описание",
                        lines=10,
                        max_lines=15
                    )
            
            with gr.Row():
                caption_button = gr.Button("Создать описание", variant="primary")
            
            # Обработчик Caption
            caption_button.click(
                fn=generate_caption,
                inputs=[caption_image],
                outputs=[caption_output]
            )
        
        # Вкладка About
        with gr.Tab("Инфо"):
            gr.Markdown(
                f"""
                ## О SmolVLM Demo
                
                Это приложение демонстрирует возможности мультимодальной модели **SmolVLM** для работы с изображениями и текстом.
                
                ### Возможности
                
                - **VQA (Visual Question Answering)**: Задавайте вопросы об изображениях и получайте интеллектуальные ответы
                - **OCR (Распознавание текста)**: Извлекайте текст из изображений
                - **Описание изображений**: Автоматическая генерация описаний для изображений
                - **Мультивопросные диалоги**: Продолжайте задавать вопросы об одном и том же изображении
                - **Экспорт результатов**: Скачивайте результаты в виде текстовых файлов
                
                ### Информация о модели
                
                - **Название модели**: `{settings.MODEL_NAME}`
                - **Размер модели**: `{settings.MODEL_SIZE}`
                - **Устройство**: `{settings.DEVICE}`
                - **Максимум токенов**: `{settings.MAX_NEW_TOKENS}`
                - **Температура**: `{settings.TEMPERATURE}`
                
                ### Доступ к API
                
                Приложение также предоставляет REST API. Посетите [/docs](/docs) для документации API.
                
                ### Ограничения
                
                - Максимальный размер изображения: {settings.MAX_IMAGE_SIZE / (1024*1024):.0f} МБ
                - Максимальное разрешение: {settings.MAX_IMAGE_DIMENSION} пикселей
                - Таймаут сессии: {settings.SESSION_TIMEOUT} секунд
                
                ### Ресурсы
                
                - [SmolVLM на HuggingFace](https://huggingface.co/HuggingFaceTB)
                - [Документация проекта](/docs)
                - [Справочник API](/docs)
                """
            )
    
    return demo

