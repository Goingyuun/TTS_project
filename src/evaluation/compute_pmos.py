import os
import pandas as pd
import soundfile as sf
import speechmetrics
import matplotlib.pyplot as plt

# 加载 MOS 评分模型
metrics = speechmetrics.load(["mosnet"], window=None)

# 读取生成的语音日志
csv_path = "results/generated_speech/generation_log.csv"
df = pd.read_csv(csv_path)

pmos_scores = []

# 遍历所有生成的音频，计算 PMOS 评分
for _, row in df.iterrows():
    audio_path = row["output_file"]
    y, sr = sf.read(audio_path)

    # 计算 MOS 评分
    mos_score = metrics(y, sr)["mosnet"]
    pmos_scores.append(mos_score[0])  # 获取平均值

df["PMOS"] = pmos_scores
df.to_csv(csv_path, index=False)

# 绘制柱状图
plt.figure(figsize=(12, 5))
plt.bar(df["sample_index"], df["PMOS"], color="skyblue")
plt.xlabel("Sample Index")
plt.ylabel("PMOS Score")
plt.title("Perceptual MOS Score Distribution")
plt.savefig("results/generated_speech/pmos_scores.png")
plt.show()
