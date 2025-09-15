import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# --- Local Module Imports ---
from config import config
from data.data_parser import parse_crema_d_data
from data.data_loader import create_data_loaders
from model.architecture import MultimodalEmotionModel
from training.trainer import Trainer
from training.evaluator import Evaluator

def main():
    """Main function to run the emotion recognition pipeline."""
    
    # 1. Load and preprocess data
    if not os.path.exists(config.DATA_CSV_PATH):
        print("Parsing CREMA-D data for the first time...")
        df = parse_crema_d_data(config.CREMA_D_AUDIO_PATH, config.CREMA_D_VIDEO_PATH)
        df.to_csv(config.DATA_CSV_PATH, index=False)
    else:
        print(f"Loading pre-parsed data from {config.DATA_CSV_PATH}...")
        df = pd.read_csv(config.DATA_CSV_PATH)
    
    df.dropna(inplace=True)
    df = df[df['emotion'].isin(config.EMOTIONS_TO_USE)].reset_index(drop=True)
    
    print(f"Using {len(df)} samples for emotions: {config.EMOTIONS_TO_USE}")

    # 2. Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    # 3. Initialize tokenizer and create DataLoaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, test_loader, emotion_map = create_data_loaders(train_df, test_df, tokenizer, config)
    idx_to_emotion = {i: emotion for emotion, i in emotion_map.items()}

    # 4. Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalEmotionModel(num_emotions=len(config.EMOTIONS_TO_USE)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training
    trainer = Trainer(model, train_loader, optimizer, criterion, device, config.EPOCHS)
    trainer.train()

    # 6. Evaluation
    class_names = [idx_to_emotion[i] for i in range(len(config.EMOTIONS_TO_USE))]
    evaluator = Evaluator(model, test_loader, device, class_names)
    evaluator.evaluate()
    
    # 7. Save the model
    print(f"\nSaving model to {config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()