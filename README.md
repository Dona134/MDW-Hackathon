# Hackathon_emotions
 
# Tweet Emotion Classification

This project was developed for the **MDW2024 Hackathon**, with the goal of classifying tweets into four emotion categories: **Anger**, **Joy**, **Sadness**, and **Optimism**. The project focuses on data preprocessing, feature extraction, and machine learning model training using **XGBoost**.

## Table of Contents

- [Project Overview](#project-overview)
- [MDW2024 Hackathon](#mdw2024-hackathon)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Model](#model)
- [Evaluation](#evaluation)
- [Files in the Project](#files-in-the-project)
- [Dependencies](#dependencies)
- [Instructions](#instructions)

## Project Overview

The objective of this project is to classify tweets into four predefined emotion categories based on text data. The classification model is trained using **TF-IDF** features and the **XGBoost** classifier, with additional attention given to class imbalance through the use of sample weights.

## MDW2024 Hackathon

This project was submitted for the **MDW2024 Hackathon**, where participants were tasked with building innovative solutions using machine learning techniques. The challenge we addressed was the emotion classification of social media posts, particularly tweets, based on their content.

## Dataset

The project uses three primary datasets:
- `train_text.txt`: Contains training tweets, one per line.
- `train_labels.txt`: Contains integer labels corresponding to the emotions of the tweets in `train_text.txt`. The labels correspond to the following emotions:
  - 0: Anger
  - 1: Joy
  - 2: Sadness
  - 3: Optimism
- `val_text.txt`: Contains validation tweets, one per line.
- `val_labels.txt`: Contains labels corresponding to `val_text.txt` (same format as `train_labels.txt`).
- `test_text.txt`: Contains the test tweets for which predictions need to be generated.

## Preprocessing Steps

1. **Emoji Conversion**: Emojis in the tweets are converted to text using the `emoji` library.
2. **Tokenization**: The text is tokenized into individual words using `nltk`.
3. **Stop Words Removal**: Common English stop words are removed using the `stopwords` list from `nltk`.
4. **Stemming**: Words are stemmed using the `SnowballStemmer` to reduce them to their root forms.
5. **Lemmatization**: Words are further reduced to their lemma form using `spaCy`.

## Model

We use **XGBoost** (`XGBClassifier`) as the primary classifier for this task. The model is trained using **TF-IDF (Term Frequency-Inverse Document Frequency)** as features. To address the class imbalance problem, **sample weights** are applied based on the frequency of each class.

### Key Model Parameters:
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 3
- **random_state**: 42

## Evaluation

The model is evaluated based on its performance on the validation set. The primary evaluation metric is the **macro-averaged F1 score**, which is computed using the predictions on the validation data.

## Files in the Project

- `train_text.txt`: Training data (tweets).
- `train_labels.txt`: Training labels (0-3 corresponding to Anger, Joy, Sadness, Optimism).
- `val_text.txt`: Validation data (tweets).
- `val_labels.txt`: Validation labels.
- `test_text.txt`: Test data (tweets) for which you will predict the labels.
- `test_predictions.csv`: Output file containing the predicted labels for the test set.
- `README.md`: This file, describing the project structure and usage.

## Dependencies

The following Python libraries are required to run the project:

- `numpy`
- `pandas`
- `nltk`
- `emoji`
- `spacy`
- `xgboost`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas nltk emoji spacy xgboost scikit-learn
```

Additionally, you will need to download the necessary NLTK data (stopwords and tokenizers) and spaCy language model:

```bash
python -m nltk.downloader stopwords punkt
python -m spacy download en_core_web_sm
```

## Instructions

### 1. Preprocess the Data

The text data is preprocessed by:
- Converting emojis to text.
- Tokenizing the text.
- Removing stop words.
- Stemming and lemmatizing words.

### 2. Train the Model

Run the Python script to train the XGBoost model using the preprocessed training data (`train_text.txt` and `train_labels.txt`). The model will then make predictions on the validation set (`val_text.txt` and `val_labels.txt`) and compute the performance metrics.

### 3. Predict and Save Results

Once the model is trained, it will predict the emotions for the test set (`test_text.txt`) and save the predictions to `test_predictions.csv`.

### 4. Evaluate Performance

The model’s performance is evaluated on the validation set using precision, recall, and the F1 score. The macro-averaged F1 score is printed for the validation set.

## Running the Project

To run the project, simply execute the Python script in your terminal:

```bash
python your_script_name.py
```

Make sure all necessary data files are in the same directory as the script.

---

This updated README includes the project context and the MDW2024 Hackathon background. Let me know if you need any additional modifications!
