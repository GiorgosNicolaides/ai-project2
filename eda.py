from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the dataset
    df = pd.read_csv("project2_dataset.csv")

    # Number of records in the dataset
    num_records = len(df)

    # Calculate the percentage of users who made a purchase
    purchase_percentage = (df['Revenue'].sum() / num_records) * 100

    # Calculate the accuracy of a model that always predicts users won't purchase
    accuracy = ((num_records - df['Revenue'].sum()) / num_records) * 100

    # Print the answers to the questions
    print("1. Number of records in the dataset:", num_records)
    print("2. Percentage of users who made a purchase:", purchase_percentage)
    print("3. Accuracy of a model always predicting users won't purchase:", accuracy)




if __name__ == "__main__":
    main()
