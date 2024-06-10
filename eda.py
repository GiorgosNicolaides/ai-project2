import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Optional: Visualize variable distributions and their relationship with the target variable
    visualize_data(df)

def visualize_data(df):
    # Set style for the plots
    sns.set(style="whitegrid")

    # Extract variable names from the DataFrame (excluding the target variable)
    variables = df.columns.drop('Revenue')

    # Split variables into correlated and uncorrelated with the target variable
    correlated_variables = ['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    uncorrelated_variables = variables.drop(correlated_variables)

    # Create plots for variables uncorrelated with the target variable
    create_plots(uncorrelated_variables, df, 'Uncorrelated Variables with Revenue')

    # Create plots for variables correlated with the target variable
    create_plots(correlated_variables, df, 'Correlated Variables with Revenue')

def create_plots(variables, df, title):
    num_plots = len(variables)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, num_rows * 3))
    fig.suptitle(title, fontsize=22)

    for var, ax in zip(variables, axes.flatten()):
        sns.boxplot(x='Revenue', y=var, data=df, ax=ax)
        ax.set_title(f'{var} vs Revenue', fontsize=14)

    # Remove empty axes if any
    for ax in axes.flatten()[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
