import os
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TTS_Dataset(Dataset):
    def __init__(self, csv_path, mel_dir):
        """
        初始化数据集，读取索引文件。
        :param csv_path: CSV 文件路径 (data/indices/index.csv)
        :param mel_dir: Mel 特征文件目录 (data/processed/mel)
        """
        self.data = pd.read_csv(csv_path)  # 读取 CSV
        self.mel_dir = mel_dir

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取一个样本
        :param idx: 样本索引
        :return: mel_spectrogram, text, duration, speaker_id
        """
        row = self.data.iloc[idx]
        
        # 读取 Mel 频谱特征
        mel_path = os.path.join(self.mel_dir, row['mel_filename'])
        mel_spectrogram = np.load(mel_path)  # 加载 Mel 频谱
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)  # 转换为 Tensor
        
        # 获取文本
        text = row['text']  # 文本信息
        
        # 获取音频时长
        duration = torch.tensor(row['duration'], dtype=torch.float32)
        
        # 说话人 ID（可以用于多说话人 TTS 任务）
        speaker_id = row['speaker_id']
        
        return mel_spectrogram, text, duration, speaker_id
"""
# 设定数据路径
csv_file = "data/indices/index.csv"
mel_directory = "data/processed/mel"

# 创建数据集
tts_dataset = TTS_Dataset(csv_file, mel_directory)

# 测试数据集
sample_mel, sample_text, sample_duration, sample_speaker = tts_dataset[0]
print("Mel Shape:", sample_mel.shape)
print("Text:", sample_text)
print("Duration:", sample_duration)
print("Speaker ID:", sample_speaker)
"""