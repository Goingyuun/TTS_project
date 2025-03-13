import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import os

# 读取数据
df = pd.read_csv("results/generated_speech/generation_log.csv")

rtf_values = []
for _, row in df.iterrows():
    audio_path = row["output_file"]
    y, sr = sf.read(audio_path)
    duration = len(y) / sr
    rtf = row["inference_time_sec"] / duration
    rtf_values.append(rtf)

df["RTF"] = rtf_values
df.to_csv("results/generated_speech/generation_log.csv", index=False)

# 保存 RTF 条形图
plt.figure(figsize=(12, 5))
plt.bar(df["sample_index"], df["RTF"], color='coral')
plt.xlabel("Sample Index")
plt.ylabel("RTF")
plt.title("Speech Synthesis Real-Time Factor (RTF)")

output_dir = "results/generated_speech/plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/rtf_scores.png", bbox_inches='tight')
plt.close()

