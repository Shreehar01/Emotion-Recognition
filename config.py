class Config:
    CREMA_D_AUDIO_PATH="./dataset/audio-files"
    CREMA_D_VIDEO_PATH="./dataset/video-files"
    DATA_CSV_PATH="./crema_d_data.csv"
    SAMPLE_RATE=16000
    N_MFCC=40
    MAX_TEXT_LEN=50
    EMOTIONS_TO_USE=['anger','disgust','fear','happy','neutral','sadness']
    FRAME_COUNT=16
    IMAGE_SIZE=224
    BATCH_SIZE=16
    EPOCHS=40
    LEARNING_RATE=1e-4
    MODEL_SAVE_PATH='multimodal_emotion_model.pth'

config=Config()
