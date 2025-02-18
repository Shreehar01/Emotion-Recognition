import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                text_ids = batch['text_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_features = batch['video_features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(text_ids, attention_mask, audio_features, video_features)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
        print("-" * 50)
        self.show_classification_report(all_labels, all_preds)
        self.show_confusion_matrix(all_labels, all_preds)

    def show_classification_report(self, labels, preds):
        print(classification_report(labels, preds, target_names=self.class_names))
        print("-" * 50)
        
    def show_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()