# NLP EN-ID Translation Using Simple RNN

This project focuses on training an English-to-Indonesian neural translation model using TensorFlow and Keras. It includes a Jupyter Notebook that demonstrates the training process on the Helsinki-NLP's English-Indonesian dataset with an RNN architecture, and evaluates the model's performance using BLEU scores.

## Features
- Loads and preprocesses English-Indonesian text data.
- Tokenizes and sequences text for both training and validation datasets.
- Defines a Recurrent Neural Network (RNN) model for translation tasks.
- Trains the model with custom callbacks, saving checkpoints and logs.
- Visualizes training loss.
- Evaluates model performance using BLEU scores.
- Provides a function for translating sample sentences.

## Dataset
The training, validation, and test data are sourced from the [Helsinki-NLP Opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) dataset in English and Indonesian.

## Python Library Requirements
- TensorFlow
- Keras
- Pandas
- Matplotlib
- NLTK

## Training

1. **Install dependencies**:
   ```bash
   pip install tensorflow pandas matplotlib nltk

2. **Run the notebook** to train the model, evaluate, and test predictions:
   - Load the data.
   - Preprocess and sequence it.
   - Train the model.
   - Evaluate using BLEU scores.

3. **Translate a Sentence**:  
   After training, use the `translate_sentence` function to translate English to Indonesian.

## Model Architecture
- Embedding Layer (64 dim)
- Simple RNN Layer (64 unit)
- Simple RNN Layer (128 unit)
- Dense Layer

## Evaluation
The model is evaluated using BLEU scores on a sample test set from datasets.

## File Structure
- `model_training.ipynb`: Main notebook for training and evaluating the model.
- `model_test.ipynb` : Notebook for testing models.
- Saved model checkpoints and logs are stored in the specified directory.
