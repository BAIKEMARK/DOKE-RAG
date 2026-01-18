# DOKE-RAG PDF Parser Dockerfile
#
# 使用方法：
# 1. 构建：docker build -t doke-rag-pdf-parser .
# 2. 运行：docker run -v $(pwd)/data:/app/data doke-rag-pdf-parser

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data/processed data/extracted_images logs models

# 设置环境变量
ENV PYTHONPATH=/app
ENV PIPELINE_MODE=strict
ENV USE_GPU=false
ENV OCR_LANG=ch

# 暴露端口（如果需要 API）
# EXPOSE 8000

# 默认命令
CMD ["python", "-c", "from doke_rag.pipeline import PDFParser; print('DOKE-RAG PDF Parser Ready!')"]
