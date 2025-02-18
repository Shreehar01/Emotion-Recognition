import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, device, epochs):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in progress_bar:
                text_ids = batch['text_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_features = batch['video_features'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(text_ids, attention_mask, audio_features, video_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")