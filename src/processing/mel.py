import os
import librosa
import numpy as np
from tqdm import tqdm

# 参数设置，根据 FastSpeech2 的常见配置
SAMPLE_RATE = 22050  # 采样率
N_FFT = 1024  # FFT 大小
HOP_LENGTH = 256  # 帧移
N_MELS = 80  # Mel 通道数

# 输入预处理后的音频目录和输出特征存放目录
AUDIO_DIR = r"D:/TTS/data/VCTK_processed"  # 之前预处理后的音频目录
OUTPUT_DIR = r"D:/TTS/data/VCTK_features"  # Mel 频谱特征保存目录


def extract_mel(
    audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
):
    """
    从音频信号中提取 Mel 频谱，并转换为对数刻度
    """
    # 提取 Mel 频谱（返回的是功率谱）
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # 转换为对数尺度（dB）
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def process_file(file_path, output_path):
    """
    加载单个音频文件，提取 Mel 频谱特征后保存为 .npy 文件
    """
    try:
        # 加载音频，保证采样率和单声道
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        # 提取 Mel 特征
        mel_feature = extract_mel(audio)
        # 保证输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 保存为 .npy 文件
        np.save(output_path, mel_feature)
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")


def process_directory(input_dir, output_dir):
    """
    递归遍历目录，对每个 .wav 文件提取 Mel 频谱特征
    """
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.lower().endswith(".wav"):
                input_file = os.path.join(root, file)
                # 构造输出文件路径，保持原目录结构，文件后缀改为 .npy
                rel_path = os.path.relpath(root, input_dir)
                output_dir_full = os.path.join(output_dir, rel_path)
                os.makedirs(output_dir_full, exist_ok=True)
                file_name, _ = os.path.splitext(file)
                output_file = os.path.join(output_dir_full, file_name + ".npy")
                process_file(input_file, output_file)


if __name__ == "__main__":
    process_directory(AUDIO_DIR, OUTPUT_DIR)
    print("特征提取完成！")
