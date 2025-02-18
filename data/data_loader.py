import cv2
import torch
import torchaudio
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CremaDDataset(Dataset):
    def __init__(self, df, tokenizer, config, emotion_to_idx=None):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        if emotion_to_idx:
            self.emotion_to_idx = emotion_to_idx
        else:
            self.emotion_to_idx = {emotion: i for i, emotion in enumerate(df['emotion'].unique())}
        self.idx_to_emotion = {i: emotion for emotion, i in self.emotion_to_idx.items()}
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            waveform, sr = torchaudio.load(row['audio_path'])
            if sr != self.config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, self.config.SAMPLE_RATE)
                waveform = resampler(waveform)
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.config.SAMPLE_RATE, n_mfcc=self.config.N_MFCC
            )
            mfcc = mfcc_transform(waveform).mean(dim=-1).squeeze()
        except Exception as e:
            mfcc = torch.zeros(self.config.N_MFCC)
        video_frames = self._load_video_frames(row['video_path'])
        encoding = self.tokenizer.encode_plus(
            row['transcription'], add_special_tokens=True, max_length=self.config.MAX_TEXT_LEN,
            return_token_type_ids=False, padding='max_length', return_attention_mask=True,
            return_tensors='pt', truncation=True
        )
        return {
            'text_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'audio_features': mfcc,
            'video_features': video_frames,
            'label': torch.tensor(self.emotion_to_idx[row['emotion']], dtype=torch.long)
        }

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        if not cap.isOpened():
            while len(frames) < self.config.FRAME_COUNT:
                frames.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            return torch.stack(frames)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.config.FRAME_COUNT, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.video_transform(frame))
        cap.release()
        while len(frames) < self.config.FRAME_COUNT:
            frames.append(torch.zeros(3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        return torch.stack(frames)


def create_data_loaders(train_df, test_df, tokenizer, config):
    train_dataset = CremaDDataset(train_df, tokenizer, config)
    test_dataset = CremaDDataset(test_df, tokenizer, config, emotion_to_idx=train_dataset.emotion_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=4)
    return train_loader, test_loader, train_dataset.emotion_to_idx