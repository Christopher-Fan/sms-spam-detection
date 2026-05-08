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

if __name__ == "__main__":
    main()