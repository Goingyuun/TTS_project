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
