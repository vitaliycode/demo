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
            history: История чата
            
        Returns:
            Кортеж из (обновленная_история, очищенный_ввод, файл_для_скачивания)
        """
        if history is None:
            history = []
        
        message = (message or "").strip()
        
        if not message:
            return history, "", None
        
        if not image:
            history.append({
                "role": "assistant",
                "content": "⚠️ Please upload an image first."
            })
            return history, "", None
        
        # Добавление сообщения пользователя
        history.append({"role": "user", "content": message})
        
        # Добавление placeholder для ассистента
        history.append({"role": "assistant", "content": "🔄 Обработка..."})
        
        try:
            # Запуск инференса
            start_time = time.time()
            answer = model_manager.vqa_inference(image, message)
            processing_time = time.time() - start_time
            
            # Обновление истории с ответом
            history[-1]["content"] = answer
            
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
            history[-1]["content"] = f"❌ Error: {str(e)}"
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
            return "⚠️ Please upload an image with text.", None
        
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
            return f"❌ Error: {str(e)}", None
    
    def generate_caption(image: Optional[Image.Image]) -> str:
        """
        Генерация подписи к изображению.
        
        Args:
            image: PIL изображение
            
        Returns:
            Текст подписи
        """
        if not image:
            return "⚠️ Please upload an image."
        
        try:
            caption = model_manager.caption_inference(image)
            return caption
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
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
    with gr.Blocks(
        title=f"{settings.APP_NAME}",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Заголовок
        gr.Markdown(
            f"""
            <div class="header-text">
            
            # 🤖 {settings.APP_NAME}
            
            **Model:** `{settings.MODEL_NAME}`  
            **Device:** `{settings.DEVICE}` | **Version:** `{settings.VERSION}`
            
            Multimodal Vision-Language Model for VQA, OCR, and Image Captioning
            
            </div>
            """
        )
        
        # Вкладка VQA
        with gr.Tab("💬 Visual Question Answering"):
            gr.Markdown(
                """
                <div class="info-box">
                📸 Upload an image and ask questions about it. The model will analyze the image and provide detailed answers.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    vqa_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    vqa_chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        type="messages"
                    )
            
            with gr.Row():
                vqa_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the image...",
                    lines=2
                )
            
            with gr.Row():
                vqa_submit = gr.Button("🚀 Send", variant="primary", scale=2)
                vqa_clear = gr.Button("🗑️ Clear Chat", scale=1)
            
            with gr.Row():
                vqa_file = gr.File(label="💾 Download Last Answer")
            
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
        with gr.Tab("📝 OCR (Text Recognition)"):
            gr.Markdown(
                """
                <div class="info-box">
                📄 Extract text from images. Upload an image containing text and get the recognized text output.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ocr_image = gr.Image(
                        label="Upload Image with Text",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    ocr_output = gr.Textbox(
                        label="Extracted Text",
                        lines=20,
                        max_lines=30
                    )
            
            with gr.Row():
                ocr_button = gr.Button("🔍 Extract Text", variant="primary")
            
            with gr.Row():
                ocr_file = gr.File(label="💾 Download Result")
            
            # Обработчик OCR
            ocr_button.click(
                fn=run_ocr,
                inputs=[ocr_image],
                outputs=[ocr_output, ocr_file]
            )
        
        # Вкладка Captioning
        with gr.Tab("🖼️ Image Captioning"):
            gr.Markdown(
                """
                <div class="info-box">
                ✨ Generate descriptive captions for your images automatically.
                </div>
                """
            )
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    caption_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=500
                    )
                
                with gr.Column(scale=1):
                    caption_output = gr.Textbox(
                        label="Generated Caption",
                        lines=10,
                        max_lines=15
                    )
            
            with gr.Row():
                caption_button = gr.Button("✨ Generate Caption", variant="primary")
            
            # Обработчик Caption
            caption_button.click(
                fn=generate_caption,
                inputs=[caption_image],
                outputs=[caption_output]
            )
        
        # Вкладка About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                f"""
                ## About SmolVLM Demo
                
                This application demonstrates the capabilities of the **SmolVLM** multimodal vision-language model.
                
                ### Features
                
                - **Visual Question Answering (VQA)**: Ask questions about images and get intelligent answers
                - **Optical Character Recognition (OCR)**: Extract text from images
                - **Image Captioning**: Generate descriptive captions automatically
                - **Multi-turn Conversations**: Continue asking questions about the same image
                - **Export Results**: Download your results as text files
                
                ### Model Information
                
                - **Model Name**: `{settings.MODEL_NAME}`
                - **Model Size**: `{settings.MODEL_SIZE}`
                - **Device**: `{settings.DEVICE}`
                - **Max Tokens**: `{settings.MAX_NEW_TOKENS}`
                - **Temperature**: `{settings.TEMPERATURE}`
                
                ### API Access
                
                This application also provides a REST API. Visit [/docs](/docs) for API documentation.
                
                ### Limitations
                
                - Maximum image size: {settings.MAX_IMAGE_SIZE / (1024*1024):.0f} MB
                - Maximum image dimension: {settings.MAX_IMAGE_DIMENSION} pixels
                - Session timeout: {settings.SESSION_TIMEOUT} seconds
                
                ### Resources
                
                - [SmolVLM on HuggingFace](https://huggingface.co/HuggingFaceTB)
                - [Project Documentation](/docs)
                - [API Reference](/docs)
                """
            )
    
    return demo

