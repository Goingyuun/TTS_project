import os
import librosa
import soundfile as sf
from tqdm import tqdm

# 配置参数
INPUT_DIR = r"D:/TTS/data/VCTK"  # 原始数据目录（包含 clean_trainset 和 clean_testset）
OUTPUT_DIR = r"D:/TTS/data/VCTK_processed"  # 处理后的数据存放目录
TARGET_SR = (
    22050  # 目标采样率，根据模型要求设置（如 FastSpeech2 常用 22050Hz 或 24000Hz）
)
MAX_DURATION = None  # 每个音频最大时长（秒），可根据需要调整；若无需限制可设为 None


def process_audio_file(in_file, out_file, sr=TARGET_SR, max_duration=MAX_DURATION):
    try:
        # 加载音频，同时重采样为目标采样率，强制转换为单声道
        audio, _ = librosa.load(in_file, sr=sr, mono=True)
        # 去除前后静音部分
        audio, _ = librosa.effects.trim(audio)
        """
        # 限制时长（如音频超过 max_duration，则截取前面的部分）
        if max_duration is not None:
            max_length = sr * max_duration
            if len(audio) > max_length:
                audio = audio[:max_length]
        """
        # 归一化音频（防止幅值过大）
        if max(abs(audio)) > 0:
            audio = audio / max(abs(audio))
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        # 保存处理后的音频文件
        sf.write(out_file, audio, sr)
    except Exception as e:
        print(f"处理文件 {in_file} 时出错: {e}")


def process_directory(input_dir, output_dir):
    # 遍历所有子目录中的 wav 文件
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.lower().endswith(".wav"):
                in_file = os.path.join(root, file)
                # 构造输出文件路径，保持原有目录结构
                rel_path = os.path.relpath(root, input_dir)
                out_dir_full = os.path.join(output_dir, rel_path)
                os.makedirs(out_dir_full, exist_ok=True)
                out_file = os.path.join(out_dir_full, file)
                process_audio_file(in_file, out_file)


if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
    print("预处理完成！")
