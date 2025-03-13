# 文件路径: src/utils/dataset.py
import os
import glob
from torch.utils.data import Dataset

class TTS_Dataset(Dataset):
    def __init__(self, txt_root="data/raw/txt"):
        """
        初始化数据集，递归加载 txt_root 下所有说话人文件夹中的文本文件。
        每个样本包含：文本、说话人ID、语句ID。
        """
        self.samples = []
        # 遍历 txt_root 下的所有子目录（例如 p226, p227, ...）
        for speaker in os.listdir(txt_root):
            speaker_folder = os.path.join(txt_root, speaker)
            if os.path.isdir(speaker_folder):
                # 对于每个子目录，找到所有 .txt 文件
                for txt_file in glob.glob(os.path.join(speaker_folder, "*.txt")):
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    # 语句ID取自文件名（不含扩展名）
                    utterance_id = os.path.splitext(os.path.basename(txt_file))[0]
                    self.samples.append({
                        "speaker_id": speaker,
                        "utterance_id": utterance_id,
                        "text": text
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 返回文本、说话人ID和语句ID
        return sample["text"], sample["speaker_id"], sample["utterance_id"]

# 下面可以添加测试代码（可选）
if __name__ == "__main__":
    dataset = TTS_Dataset("data/raw/txt")
    print("样本总数：", len(dataset))
    for i in range(min(5, len(dataset))):
        text, speaker, utt = dataset[i]
        print(f"样本{i}: 说话人 {speaker}, 语句ID {utt}, 文本：{text}")

"""
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
        
        self.data = pd.read_csv(csv_path)  # 读取 CSV
        self.mel_dir = mel_dir
        """
        self.data = []
        with open(csv_path, 'r') as file:
            for line in file:
                line = line.strip().split(',')
                text = line[1]
                self.data.append(text)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # 可以添加额外的处理逻辑，如计算发音时长、说话人ID等
        durations = [len(text)]  # 示例：假设每个文本的发音时长与文本长度成正比
        speaker_id = 0  # 示例：假设所有语音来自同一说话人
        return text, durations, speaker_id
        """
        获取一个样本
        :param idx: 样本索引
        :return: mel_spectrogram, text, duration, speaker_id
        
        row = self.data.iloc[idx]
        
        # 读取 Mel 频谱特征
        mel_path = os.path.join(self.mel_dir, row['mel_filename'])
        mel_spectrogram = np.load(mel_path)  # 加载 Mel 频谱
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)  # 转换为 Tensor
         # **确保 Mel 频谱的形状一致**
        if mel_spectrogram.dim() == 2:
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # 添加通道维度 (C=1)
        # 获取文本
        text = row['text']  # 文本信息
        
        # 获取音频时长
        duration = torch.tensor(row['duration'], dtype=torch.float32)
        
        # 说话人 ID（可以用于多说话人 TTS 任务）
        speaker_id = row['speaker_id']
        
        return mel_spectrogram, text, duration, speaker_id
        """
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
"""
