import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)

# Pull in all of the results csv files
csv_files = ["results.csv", "results2.csv", "avg_results.csv", "avg_results2.csv", "avg_results3.csv", "vote_results.csv", "vote_results2.csv"]
column_titles = ['beta', 'gamma', 'T', 'best_acc', 'acc']
avg_dfs = []
vote_dfs = []
dfs = []
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Rename the columns to the desired titles
    df.columns = column_titles
    # Sort the DataFrame by Method
    if "avg" in csv_file:
        avg_dfs.append(df)
    elif "vote" in csv_file:
        vote_dfs.append(df)
    else:
        dfs.append(df)

# Combine the DataFrames into a single DataFrame
avg_df = pd.concat(avg_dfs, ignore_index=True)
vote_df = pd.concat(vote_dfs, ignore_index=True)
df = pd.concat(dfs, ignore_index=True)

print("Max accuracy for df: ", df['acc'].max())
print("Max accuracy for avg_df: ", avg_df['acc'].max())
print("Max accuracy for vote_df: ", vote_df['acc'].max())

def plot_results(df):
    # Get all the unique values of beta, gamma, and T
    betas = df['beta'].unique()
    gammas = df['gamma'].unique()
    Ts = df['T'].unique()
    print("betas: ", betas)
    print("gammas: ", gammas)
    print("Ts: ", Ts)

    # Convert negative accuracies to 0
    df['acc'] = df['acc'].apply(lambda x: max(x, 0))

    # Make hyperparameter plots
    for T_val in Ts:
        plt.figure()
        df_T0 = df[df['T'] == T_val]
        for gamma in gammas:
            df_gamma = df_T0[df_T0['gamma'] == gamma]
            plt.plot(df_gamma['beta'], df_gamma['acc'], label=gamma)
        plt.xlabel('Beta')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Beta for T = ' + str(T_val))
        plt.legend(title='Gamma')
    plt.show()

# plot_results(df) # Best beta ~ 2, 2.2 gamma ~ 0.6, T ~ 100, 2
# plot_results(avg_df) # Best beta ~ 2, gamma ~ 0.6, 0.8 T ~ 100, 2
# plot_results(vote_df) # Best beta ~ 2, gamma ~ 0.6 T ~ 2 insufficient data