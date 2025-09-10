# Speech Emotion Recognition

This project aims to recognize human emotions from speech using machine learning. The model is trained on the TESS (Toronto Emotional Speech Set) dataset to classify emotions into seven categories: fear, angry, disgust, neutral, sad, pleasant surprise, and happy.

## Dataset

The project utilizes the [TESS (Toronto Emotional Speech Set) dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess). This dataset contains a set of 200 target words spoken by two actresses (one younger, one older) in the seven emotions mentioned above. The dataset is organized into folders for each emotion.

## Exploratory Data Analysis (EDA)

The notebook includes an exploratory data analysis section where audio waveforms and spectrograms of different emotions are visualized. This helps in understanding the characteristics of the audio data for each emotion.

## Feature Extraction

To train the model, various audio features are extracted from the speech signals. These features include:

* **Mel-Frequency Cepstral Coefficients (MFCCs)**
* **Mel Spectrogram**

These features capture the essential characteristics of the speech signals that are relevant for emotion recognition.

## Model

A Convolutional Neural Network (CNN) model is built and trained to classify the emotions. The model architecture is designed to effectively learn the patterns from the extracted audio features.

## Results

The model is trained and evaluated, achieving high accuracy in classifying the emotions from the speech data. The training and validation accuracy and loss are plotted to visualize the model's performance over epochs.

## Usage

To run this project, you will need a Python environment with the required dependencies installed. You can then run the `speech_emotion_recognition.ipynb` notebook.

## Dependencies

The following Python libraries are required to run the notebook:

* pandas
* numpy
* os
* seaborn
* matplotlib
* librosa
* IPython
* scikit-learn
* tensorflow
