import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

from config import config

class MultimodalEmotionModel(nn.Module):
    """ A trimodal model for emotion recognition using text, audio, and video. """
    def __init__(self, num_emotions):
        super(MultimodalEmotionModel, self).__init__()
        # Text Encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Audio Encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(config.N_MFCC, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        
        # Video Encoder (using a pre-trained ResNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.video_encoder = nn.Sequential(*list(resnet.children())[:-1])
        video_feature_size = resnet.fc.in_features
        
        for param in self.video_encoder.parameters():
            param.requires_grad = False

        # Fusion and Classifier
        self.fusion = nn.Linear(self.bert.config.hidden_size + 64 + video_feature_size, 256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions)
        )

    def forward(self, text_ids, attention_mask, audio_features, video_features):
        # Process Text
        bert_output = self.bert(input_ids=text_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output
        
        # Process Audio
        audio_features = self.audio_encoder(audio_features)
        
        # Process Video
        b, t, c, h, w = video_features.size()
        video_features = video_features.view(b * t, c, h, w)
        video_features = self.video_encoder(video_features)
        video_features = video_features.view(b, t, -1)
        video_features = video_features.mean(dim=1)
        
        # Combine features
        combined_features = torch.cat((text_features, audio_features, video_features), dim=1)
        
        fused = torch.relu(self.fusion(combined_features))
        logits = self.classifier(fused)
        return logits