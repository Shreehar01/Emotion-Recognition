import os
import pandas as pd
from tqdm import tqdm

def parse_crema_d_data(audio_path, video_path):
    emotion_map = {
        'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sadness'
    }
    sentence_map = {
        'IEO': "It's eleven o'clock", 'TIE': "That is exactly what happened",
        'IOM': "I'm on my way to the meeting", 'IWW': "I wonder what this is about",
        'TAI': "The airplane is almost full", 'MTI': "Maybe tomorrow it will be cold",
        'IWL': "I would like a new alarm clock", 'ITH': "I think I have a doctor's appointment",
        'DFA': "Don't forget a jacket", 'ITS': "I think I've seen this before",
        'TSI': "The surface is slick", 'WSI': "We'll stop in a couple of minutes"
    }
    data = []
    for entry in tqdm(os.scandir(audio_path), desc="Parsing CREMA-D"):
        if entry.is_file() and entry.name.endswith('.wav'):
            parts = entry.name.split('_')
            if len(parts) == 4:
                _, sentence_code, emotion_code, _ = parts
                video_file = entry.name.replace('.wav', '.flv')
                full_video_path = os.path.join(video_path, video_file)
                if emotion_code in emotion_map and sentence_code in sentence_map and os.path.exists(full_video_path):
                    data.append({
                        "audio_path": entry.path,
                        "video_path": full_video_path,
                        "emotion": emotion_map[emotion_code],
                        "transcription": sentence_map[sentence_code]
                    })
    return pd.DataFrame(data)