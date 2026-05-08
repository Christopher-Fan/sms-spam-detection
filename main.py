import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_hub as hub
import random

def compile_and_fit(model, x_train_np, y_train_np, x_test_np, y_test_np, epochs=5):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train_np,
        y_train_np,
        epochs=epochs,
        validation_data=(x_test_np, y_test_np)
    )
    return history

def get_metrics(model, X, y):
    y_preds = np.round(model.predict(X))
    return {
        'accuracy': accuracy_score(y, y_preds),
        'precision': precision_score(y, y_preds),
        'recall': recall_score(y, y_preds),
        'f1-score': f1_score(y, y_preds)
    }

def main():
    #load the dataset into a df
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print(df.head())

    #clean the dataframe
    #replace unused columns
    #encoding labels to better reflect the data
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df['label_enc'] = df['label'].map({'ham':0, 'spam': 1})
    print(df.head())

    #split data into training and testing splits
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'],
        df['label_enc'],
        test_size=0.2,
        random_state=random.randint(1,1000)
        )
    
    #convert from pandas to numpy arrays for performance
    x_train_np = x_train.to_numpy()
    x_test_np = x_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    #vectorizes the messages in order to find the average message length for use in padding 
    avg_sms_len = round(sum([len(i.split()) for i in df['text']]) / len(df['text']))
    total_sms_len = len(set(" ".join(df['text']).split()))

    print(f"Data Loaded. Training samples: {len(x_train_np)}")
    print(f"Average wrod per message: {avg_sms_len}")
    print(f"Approximate vocabulary size: {total_sms_len}")

    

if __name__ == "__main__":
    main()