import os
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from nisqa.NISQA_model import nisqaModel

# 加载 NISQA 预训练模型
model = nisqaModel(mode="predict", pre_trained_model="nisqa_tar.pt")

# 读取语音生成日志
csv_path = "results/generated_speech/generation_log.csv"
df = pd.read_csv(csv_path)

pmos_scores = []

# 遍历所有生成的音频，计算 PMOS 评分
for _, row in df.iterrows():
    audio_path = row["output_file"]
    score = model.predict(audio_path)  # NISQA 预测
    pmos_scores.append(score["mos_pred"][0])  # 获取 MOS 预测值

df["PMOS"] = pmos_scores
df.to_csv(csv_path, index=False)

# 绘制 PMOS 评分分布柱状图
plt.figure(figsize=(12, 5))
plt.bar(df["sample_index"], df["PMOS"], color='skyblue')
plt.xlabel("Sample Index")
plt.ylabel("PMOS Score")
plt.title("Perceptual MOS Score Distribution")

output_dir = "results/generated_speech/plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/pmos_scores.png", bbox_inches='tight')
plt.close()
