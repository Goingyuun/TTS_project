import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def plot_mel_spectrogram(audio_path, save_path):
    """ 读取音频文件并绘制 Mel 频谱图，保存到文件 """
    y, sr = sf.read(audio_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {os.path.basename(audio_path)}')
    
    plt.savefig(save_path, bbox_inches='tight')  # 保存文件
    plt.close()

# 批量处理
output_dir = "results/generated_speech/plots/mel_spectrograms"
os.makedirs(output_dir, exist_ok=True)

for wav_file in os.listdir("results/generated_speech"):
    if wav_file.endswith(".wav"):
        plot_mel_spectrogram(
            audio_path=f"results/generated_speech/{wav_file}",
            save_path=f"{output_dir}/{wav_file.replace('.wav', '.png')}"
        )
