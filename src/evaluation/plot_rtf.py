import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

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

# 绘制 RTF 条形图
plt.figure(figsize=(12, 5))
plt.bar(df["sample_index"], df["RTF"], color="coral")
plt.xlabel("Sample Index")
plt.ylabel("RTF")
plt.title("Speech Synthesis Real-Time Factor (RTF)")
plt.savefig("results/generated_speech/rtf_scores.png")
plt.show()
