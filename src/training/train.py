import sys
import os
import time
import csv
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 加载 SpeechT5 模型及对应的声码器
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# 加载数据集（使用自定义的文本数据加载器）
from src.utils.data_loader import get_text_dataloader
dataloader = get_text_dataloader(txt_root="data/raw/txt", batch_size=1, shuffle=False)

# 指定生成语音的保存目录
output_dir = "results/generated_speech"
os.makedirs(output_dir, exist_ok=True)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
fixed_speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)  # 选取第一个说话人的嵌入
# CSV 文件记录生成的样本信息（包括文本、说话人、utterance、生成时长等）
csv_output = os.path.join(output_dir, "generation_log.csv")
with open(csv_output, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入表头
    csv_writer.writerow(["sample_index", "speaker_id", "utterance_id", "text", "inference_time_sec", "output_file"])

    # 遍历数据集，对每个文本生成语音，并保存文件
    for i, (text, speaker_id, utterance_id) in enumerate(dataloader):
        # dataloader 的输出如果 batch_size=1，会是列表，我们取第一个
        text = text[0] if isinstance(text, list) else text
        speaker_id = speaker_id[0] if isinstance(speaker_id, list) else speaker_id
        utterance_id = utterance_id[0] if isinstance(utterance_id, list) else utterance_id

        # 记录开始时间
        start_time = time.time()
        
        # 预处理文本，生成模型输入
        inputs = processor(text=text, return_tensors="pt")
        # 此处如果有自定义 speaker embedding 可以加载并传入，此处先设为 None
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=fixed_speaker_embedding, vocoder=vocoder)
        
        # 记录推理结束时间
        end_time = time.time()
        inference_time = end_time - start_time

        # 构造输出文件名，例如 "p226_p226_001.wav"
        output_filename = f"{speaker_id}_{utterance_id}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # 保存音频，采样率这里设为16000，根据模型文档调整
        sf.write(output_path, speech.numpy(), samplerate=16000)
        print(f"样本 {i}: 生成的语音保存为 {output_path}，推理耗时 {inference_time:.2f} 秒")

        # 将生成信息写入 CSV
        csv_writer.writerow([i, speaker_id, utterance_id, text, inference_time, output_path])

print(f"所有生成样本信息已保存到 {csv_output}")
