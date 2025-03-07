# TTS_Project: 智能语音识别与合成系统

本项目基于 FastSpeech2 和 HiFi-GAN 模型，设计并实现了一种智能语音识别合成系统。系统能够通过对语音个性化特征与情感的理解，实现针对不同语料的个性化语音合成，同时具备较好的泛化能力，适用于不同语音风格和语境。

## 目录

- [项目概述](#项目概述)
- [文件结构](#文件结构)
- [数据说明](#数据说明)
- [预处理和特征提取](#预处理和特征提取)
- [模型训练与评估](#模型训练与评估)
- [环境依赖](#环境依赖)
- [使用说明](#使用说明)
- [引用](#引用)
- [许可证](#许可证)

## 项目概述

本项目的主要目标包括：
- 构建数据集，提取多说话人语音及情感特征；
- 利用 FastSpeech2+HiFi-GAN 模型进行个性化语音合成；
- 设计实验和评估指标，对合成语音效果进行可视化分析。

## 文件结构

```plaintext
TTS_Project/
├── data/
│   ├── raw/                      # 原始数据
│   ├── processed/                # 预处理后的数据
│   │   ├── audio/                # 预处理后的音频文件
│   │   └── mel/                  # 提取的 Mel 频谱特征 (.npy)
│   └── indices/                  # 数据索引文件 (CSV/JSON)
├── docs/                         # 项目文档与实验报告
├── notebooks/                    
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py         # 音频预处理代码
│   │   └── extract_features.py   # 特征提取代码
│   ├── training/
│   │   ├── train.py              # 模型训练代码
│   │   └── evaluate.py           # 模型评估代码
│   ├── models/
│   │   ├── FastSpeech2.py        # FastSpeech2 模型代码
│   │   └── HiFiGAN.py            # HiFi-GAN 模型代码
│   └── utils/
│       └── data_utils.py         # 辅助工具代码
├── .gitignore                    # Git 忽略文件
├── requirements.txt              # 项目依赖包列表
└── README.md                     # 项目说明文件
```
## 数据说明
### 数据来源：
本项目使用了 VCTK 数据集 的 clean 部分，数据经过预处理后存储在 data/processed 文件夹中。

### 数据内容：

data/processed/audio/：预处理后的音频文件（统一采样率、单声道、截断处理等）。
data/processed/mel/：提取的 Mel 频谱特征文件，格式为 .npy。
data/indices/：数据索引文件（例如 CSV），记录了每个音频文件对应的说话人、文本、时长等信息。
## 预处理和特征提取
### 预处理流程：
预处理代码位于 src/preprocessing/preprocess.py，主要完成：

音频文件的重采样、静音去除、时长截断与归一化；
保持原目录结构，输出预处理后的音频文件至 data/processed/audio/。
### 特征提取：
特征提取代码位于 src/preprocessing/extract_features.py，主要完成：

通过 librosa 提取 Mel 频谱特征，并转换为对数刻度；
输出特征文件（.npy 格式）至 data/processed/mel/。
## 模型训练与评估
### 模型结构：
模型代码存放于 src/models/，包括 FastSpeech2 和 HiFi-GAN 模型的实现。

### 训练与评估：
训练代码位于 src/training/train.py，评估代码位于 src/training/evaluate.py，具体训练步骤和参数设置请参阅代码注释。

## 环境依赖
本项目主要依赖以下 Python 库：

Python 3.9
librosa
numpy
soundfile
tqdm
PyTorch
其他依赖请参见 requirements.txt
## 使用说明
1. 数据下载与预处理
2. 模型训练
3. 模型评估
4. 运行
## 引用
如果你在研究或论文中使用了本项目代码，请引用：
```latex
@misc{YourProject2025,
  author = {Your Name},
  title = {TTS_Project: 智能语音识别与合成系统},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your_username/TTS_Project}}
}
```
## 许可证
本项目采用 MIT License 进行开源，详见 LICENSE 文件。
