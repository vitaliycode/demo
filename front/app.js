// SmolVLM Demo Frontend JavaScript

class SmolVLMApp {
    constructor() {
        this.sessionId = null;
        this.ocrTaskId = null;
        this.init();
    }

    init() {
        this.setupTabs();
        this.setupImageUploads();
        this.setupEventListeners();
        this.checkHealth();
        this.loadTechDetails();
    }

    // Навигация по вкладкам
    setupTabs() {
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.getAttribute('data-tab');
                
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(tc => tc.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(`${tabId}-tab`).classList.add('active');
            });
        });
    }

    // Настройка загрузки изображений
    setupImageUploads() {
        this.setupImageUpload('vqa');
        this.setupImageUpload('ocr');
        this.setupImageUpload('caption');
    }

    setupImageUpload(prefix) {
        const input = document.getElementById(`${prefix}ImageInput`);
        const uploadArea = document.getElementById(`${prefix}UploadArea`);
        const preview = document.getElementById(`${prefix}ImagePreview`);

        uploadArea.addEventListener('click', () => input.click());

        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.handleImageFile(e.target.files[0], uploadArea, preview);
            }
        });

        // Перетаскивание файлов
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                input.files = e.dataTransfer.files;
                this.handleImageFile(e.dataTransfer.files[0], uploadArea, preview);
            }
        });
    }

    handleImageFile(file, uploadArea, preview) {
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select an image file', 'error');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            this.showToast('Image must be less than 10MB', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            uploadArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Обработчики событий
    setupEventListeners() {
        // VQA
        document.getElementById('vqaSubmit').addEventListener('click', () => this.submitVQA());
        document.getElementById('vqaQuestion').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.submitVQA();
            }
        });

        // OCR
        document.getElementById('ocrSubmit').addEventListener('click', () => this.submitOCR());
        document.getElementById('ocrDownload').addEventListener('click', () => this.downloadOCR());

        // Caption
        document.getElementById('captionSubmit').addEventListener('click', () => this.submitCaption());
    }

    // API вызовы
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const statusDot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            
            if (data.model_loaded) {
                statusDot.classList.add('healthy');
                statusText.textContent = 'Ready';
            } else {
                statusText.textContent = 'Loading model...';
                setTimeout(() => this.checkHealth(), 3000);
            }
        } catch (error) {
            const statusText = document.querySelector('.status-text');
            statusText.textContent = 'Error';
            console.error('Health check failed:', error);
        }
    }

    async loadTechDetails() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const techDetails = document.getElementById('techDetails');
            techDetails.innerHTML = `
                <p><strong>Model:</strong> ${data.model_name}</p>
                <p><strong>Device:</strong> ${data.device}</p>
                <p><strong>Status:</strong> ${data.status}</p>
                <p><strong>Version:</strong> ${data.version}</p>
            `;
        } catch (error) {
            console.error('Failed to load tech details:', error);
        }
    }

    async submitVQA() {
        const input = document.getElementById('vqaImageInput');
        const question = document.getElementById('vqaQuestion').value.trim();
        const container = document.getElementById('vqaChatContainer');
        const submitBtn = document.getElementById('vqaSubmit');

        if (!input.files || !input.files[0]) {
            this.showToast('Please upload an image first', 'error');
            return;
        }

        if (!question) {
            this.showToast('Please enter a question', 'error');
            return;
        }

        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        try {
            const base64Image = await this.fileToBase64(input.files[0]);
            
            const response = await fetch('/api/vqa', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image,
                    question: question,
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error('VQA request failed');
            }

            const data = await response.json();
            this.sessionId = data.session_id;

            // Clear empty state if present
            if (container.querySelector('.empty-state')) {
                container.innerHTML = '';
            }

            // Добавление сообщения пользователя
            this.addChatMessage(container, question, 'user');
            
            // Добавление сообщения ассистента
            this.addChatMessage(container, data.answer, 'assistant');

            // Очистка поля ввода
            document.getElementById('vqaQuestion').value = '';

            // Показ времени обработки
            if (data.processing_time) {
                this.showToast(`Обработано за ${data.processing_time.toFixed(2)}с`, 'success');
            }

        } catch (error) {
            console.error('VQA ошибка:', error);
            this.showToast('Не удалось получить ответ. Попробуйте снова.', 'error');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    }

    async submitOCR() {
        const input = document.getElementById('ocrImageInput');
        const resultArea = document.getElementById('ocrResult');
        const downloadBtn = document.getElementById('ocrDownload');
        const submitBtn = document.getElementById('ocrSubmit');

        if (!input.files || !input.files[0]) {
            this.showToast('Please upload an image first', 'error');
            return;
        }

        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        this.showLoading();

        try {
            const base64Image = await this.fileToBase64(input.files[0]);
            
            const response = await fetch('/api/ocr', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image,
                    language: 'auto'
                })
            });

            if (!response.ok) {
                throw new Error('OCR request failed');
            }

            const data = await response.json();
            
            resultArea.value = data.text;
            this.ocrTaskId = data.task_id;
            downloadBtn.disabled = false;

            if (data.processing_time) {
                this.showToast(`OCR completed in ${data.processing_time.toFixed(2)}s`, 'success');
            }

        } catch (error) {
            console.error('OCR error:', error);
            this.showToast('Failed to extract text. Please try again.', 'error');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
            this.hideLoading();
        }
    }

    async submitCaption() {
        const input = document.getElementById('captionImageInput');
        const resultDiv = document.getElementById('captionResult');
        const submitBtn = document.getElementById('captionSubmit');

        if (!input.files || !input.files[0]) {
            this.showToast('Please upload an image first', 'error');
            return;
        }

        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        this.showLoading();

        try {
            const base64Image = await this.fileToBase64(input.files[0]);
            
            const response = await fetch('/api/caption', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image
                })
            });

            if (!response.ok) {
                throw new Error('Caption request failed');
            }

            const data = await response.json();
            
            resultDiv.innerHTML = `<div class="caption-text">${this.escapeHtml(data.caption)}</div>`;

            if (data.processing_time) {
                this.showToast(`Caption generated in ${data.processing_time.toFixed(2)}s`, 'success');
            }

        } catch (error) {
            console.error('Caption error:', error);
            this.showToast('Failed to generate caption. Please try again.', 'error');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
            this.hideLoading();
        }
    }

    async downloadOCR() {
        if (!this.ocrTaskId) {
            this.showToast('No OCR result to download', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/download/ocr/${this.ocrTaskId}`);
            
            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ocr_result_${this.ocrTaskId}.txt`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.showToast('Download started', 'success');
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('Failed to download result', 'error');
        }
    }

    // Вспомогательные методы
    addChatMessage(container, content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = role === 'user' ? 'Вы' : 'Ассистент';
        
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${role}`;
        bubble.textContent = content;
        
        messageDiv.appendChild(label);
        messageDiv.appendChild(bubble);
        container.appendChild(messageDiv);
        
        // Прокрутка вниз
        container.scrollTop = container.scrollHeight;
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showLoading() {
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const messageSpan = toast.querySelector('.toast-message');
        
        messageSpan.textContent = message;
        toast.className = `toast ${type}`;
        toast.classList.remove('hidden');
        
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }
}

// Инициализация приложения когда DOM готов
document.addEventListener('DOMContentLoaded', () => {
    new SmolVLMApp();
});

