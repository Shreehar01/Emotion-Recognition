## Multimodal Emotion Recognition from Video, Audio, and Text
This project implements a trimodal deep learning model to recognize human emotions from video, audio, and text data. It uses the CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) and leverages pre-trained models like BERT and ResNet for feature extraction, combining them to make a final emotion classification.

This project was developed as an exploration into Multimodal Learning.    

### Getting Started
Follow these instructions to set up the environment, prepare the data, and run the code.

#### Prerequisites
Python 3.8+

Git

pip (Python package installer)

### 1. Installation
First, clone the repository containing the Python script and navigate into the project directory.

Next, install the required Python libraries. Run the following command in your terminal to install them:

```
pip install -r requirements.txt
```

### 2. Data Preparation
This project uses the CREMA-D dataset(https://github.com/CheyneyComputerScience/CREMA-D)

After downloading and unzipping the dataset, you need to organize the files as follows:

From the downloaded CREMA-D AudioWAV folder, copy all the .wav files into your audio-files directory.

From the downloaded CREMA-D VideoFLV folder, copy all the .flv files into your video-files directory.

Your final project directory structure should look like this:

```
/project-root
|-- audio-files/
|   |-- 1001_IEO_ANG_HI.wav
|   |-- 1001_IEO_DIS_HI.wav
|   |-- ... (all other .wav files)
|-- video-files/
|   |-- 1001_IEO_ANG_HI.flv
|   |-- 1001_IEO_DIS_HI.flv
|   |-- ... (all other .flv files)
|-- *.py
|-- app.py
|-- requirements.txt
```

### 3. Running the Script
Once the dependencies are installed and the data is in place, you can run the script from your terminal:

```
python app.py
```

The script will first parse the filenames to create a CSV file mapping audio, video, text, and emotion labels. It will then proceed to train the model, evaluate it on the test set, and print the results, including a classification report and a confusion matrix. Finally, it will save the trained model weights to a file named multimodal_emotion_model.pth.


