class Config:
    """
    Configuration class for the Multimodal Emotion Recognition project.
    Contains all configurable parameters for data, model, and training.
    """
    # --- Dataset Paths ---
    # IMPORTANT: Update these paths to your downloaded CREMA-D dataset locations
    CREMA_D_AUDIO_PATH = "./dataset/audio-files" 
    CREMA_D_VIDEO_PATH = "./dataset/video-files"
    DATA_CSV_PATH = "./crema_d_data.csv" # Path to save the parsed dataframe

    # --- Data Preprocessing ---
    SAMPLE_RATE = 16000
    N_MFCC = 40
    MAX_TEXT_LEN = 50
    EMOTIONS_TO_USE = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness']
    
    # --- Video Settings ---
    FRAME_COUNT = 16  # Number of frames to sample from each video
    IMAGE_SIZE = 224  # Resize frames to this size
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 16
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    
    # --- Model Saving ---
    MODEL_SAVE_PATH = 'multimodal_emotion_model.pth'

# Instantiate config
config = Config()