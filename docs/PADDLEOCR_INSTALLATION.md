# PaddleOCR 安装指南

本指南详细说明如何安装和配置 PaddleOCR，用于 DOKE-RAG 的 PDF 解析功能。

## 目录

- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [验证安装](#验证安装)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

---

## 系统要求

### 基本要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows / Linux / macOS
- **内存**: 至少 4GB RAM（推荐 8GB+）
- **磁盘空间**: 至少 2GB（用于模型文件）

### GPU 加速（可选）

如果需要使用 GPU 加速：

- **CUDA**: 11.2 或更高版本
- **cuDNN**: 8.0 或更高版本
- **GPU**: NVIDIA GPU，计算能力 6.0+（推荐 RTX 3060 或更高）

---

## 安装步骤

### 方法 1：使用 pip 安装（推荐）

#### 1.1 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

#### 1.2 安装 PaddlePaddle

**CPU 版本**（默认）：

```bash
pip install paddlepaddle>=2.5.0
```

**GPU 版本**（需要 CUDA）：

```bash
pip install paddlepaddle-gpu>=2.5.0
```

#### 1.3 安装 PaddleOCR

```bash
pip install paddleocr>=2.7.0
```

#### 1.4 安装其他依赖

```bash
# PDF 处理
pip install pdfplumber>=0.10.0

# 图像处理
pip install pillow>=10.0.0
pip install opencv-python>=4.8.0

# 表格处理
pip install html2text>=2020.1.16
pip install beautifulsoup4>=4.12.0
```

#### 1.5 安装 DOKE-RAG 依赖

```bash
# 在项目根目录
pip install -r requirements.txt
```

### 方法 2：使用 conda 安装

```bash
# 创建 conda 环境
conda create -n doke-rag python=3.9
conda activate doke-rag

# 安装 PaddlePaddle
# CPU 版本:
conda install paddlepaddle -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

# GPU 版本:
conda install paddlepaddle-gpu -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

# 安装其他依赖
pip install -r requirements.txt
```

---

## 验证安装

### 1. 验证 PaddlePaddle

```bash
python -c "import paddle; print(paddle.__version__); print(paddle.device.get_device())"
```

预期输出：
```
2.5.0  # 或更高版本
cpu     # 或 gpu
```

### 2. 验证 PaddleOCR

```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='ch'); print('PaddleOCR initialized successfully')"
```

首次运行会自动下载模型文件（约 10-20MB），请耐心等待。

### 3. 测试 OCR 功能

创建测试脚本 `test_ocr.py`：

```python
from paddleocr import PaddleOCR
import cv2

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 测试图片路径
img_path = 'path/to/test_image.jpg'

# 执行识别
result = ocr.ocr(img_path, cls=True)

# 打印结果
for line in result:
    print(line)
```

### 4. 运行 DOKE-RAG 测试

```bash
# 创建测试 PDF
cd tests/pipeline
python create_test_pdf.py

# 运行测试
cd ../../
pytest tests/pipeline/test_pdf_parser.py -v
```

---

## 常见问题

### 问题 1：ImportError: DLL load failed

**症状**：
```
ImportError: DLL load failed while importing paddle
```

**解决方案**：

1. 确保 Visual C++ Redistributable 已安装
2. 重新安装 PaddlePaddle：
   ```bash
   pip uninstall paddlepaddle
   pip install paddlepaddle
   ```

### 问题 2：模型下载失败

**症状**：
```
Failed to download model from https://paddleocr.bj.bcebos.com/...
```

**解决方案**：

1. 使用国内镜像：
   ```bash
   export HUB_URL=https://hub.paddlepaddle.org.cn
   ```

2. 手动下载模型：
   ```bash
   # 创建模型目录
   mkdir -p models/paddleocr/det
   mkdir -p models/paddleocr/rec
   mkdir -p models/paddleocr/cls

   # 下载模型（从 PaddleOCR GitHub 仓库）
   # 将模型文件放到对应目录
   ```

### 问题 3：GPU 不可用

**症状**：
```
 paddle.device.set_device('gpu') 报错
```

**解决方案**：

1. 检查 CUDA 安装：
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. 确认安装了 GPU 版本：
   ```bash
   pip list | grep paddle
   # 应该显示 paddlepaddle-gpu
   ```

3. 检查 cuDNN 版本是否匹配

### 问题 4：pdfplumber 报错

**症状**：
```
AttributeError: 'PDFPage' object has no attribute 'extract_text'
```

**解决方案**：

```bash
pip uninstall pdfplumber
pip install pdfplumber>=0.10.0
```

### 问题 5：内存不足

**症状**：
```
OutOfMemoryError 或系统卡死
```

**解决方案**：

1. 处理较小的 PDF 文件
2. 逐页处理而不是批量处理
3. 减少并发数：
   ```python
   # 在配置中设置
   LIMIT_PARALLELISM = 1
   ```

---

## 性能优化

### 1. 使用 GPU 加速

```bash
# 1. 卸载 CPU 版本
pip uninstall paddlepaddle

# 2. 安装 GPU 版本
pip install paddlepaddle-gpu

# 3. 在 .env 中配置
USE_GPU=true
```

### 2. 调整批处理大小

```python
# 在 pdf_parser.py 中调整
self.ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    # 增加批处理大小（需要更多内存）
    max_batch_size=10  # 默认是 1
)
```

### 3. 模型量化

```bash
# 使用量化模型（精度略降，速度提升）
# 在初始化时指定
ocr = PaddleOCR(
    det_model_dir=path_to_quantized_det_model,
    rec_model_dir=path_to_quantized_rec_model
)
```

### 4. 多进程处理

```python
# 对于大量 PDF 文件，可以使用多进程
from multiprocessing import Pool

def process_single_pdf(pdf_path):
    parser = PDFParser()
    return await parser.parse_pdf(pdf_path, "output/")

# 使用进程池
with Pool(processes=4) as pool:
    results = pool.map(process_single_pdf, pdf_files)
```

---

## 配置选项

在 `.env` 文件中可以配置以下参数：

```bash
# PaddleOCR 模型目录（自动下载）
PADDLE_MODEL_DIR=models/paddleocr/

# 是否提取图片
EXTRACT_IMAGES=true

# 图片输出目录
IMAGES_OUTPUT_DIR=data/extracted_images/

# 公式格式（latex 或 image）
FORMULA_FORMAT=latex

# 最大 PDF 文件大小（MB）
MAX_PDF_SIZE=100

# 是否使用 GPU
USE_GPU=false

# OCR 语言（ch-中文, en-英文）
OCR_LANG=ch

# 运行模式（strict-严格, tolerant-容错）
PIPELINE_MODE=strict
```

---

## 卸载

如果需要卸载 PaddleOCR：

```bash
pip uninstall paddleocr paddlepaddle
```

删除模型文件：

```bash
rm -rf ~/.paddleocr/
rm -rf models/paddleocr/
```

---

## 参考资源

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle 官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/index.html)
- [PaddleOCR 模型库](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md)

---

## 技术支持

如果遇到问题：

1. 查看上述常见问题
2. 搜索 [PaddleOCR Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)
3. 在 DOKE-RAG 仓库提出 Issue
