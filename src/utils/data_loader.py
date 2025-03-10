from torch.utils.data import DataLoader
from .dataset import TTS_Dataset  # 引入数据集类

def collate_fn(batch):
    """
    处理不同长度的文本数据
    """
    mel_spectrograms, texts, durations, speakers = zip(*batch)
    mel_spectrograms = torch.nn.utils.rnn.pad_sequence(mel_spectrograms, batch_first=True, padding_value=0)
    durations = torch.tensor(durations, dtype=torch.float32)
    return mel_spectrograms, texts, durations, speakers

def get_dataloader(csv_path, mel_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = TTSDataset(csv_path, mel_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader
