import os

os.system("python src/evaluation/plot_mel_spectrogram.py")
os.system("python src/evaluation/compute_pmos.py")
os.system("python src/evaluation/plot_rtf.py")

print("✅ 所有图表已生成，保存于 results/generated_speech/plots/ 目录")
