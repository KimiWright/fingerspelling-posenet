import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def csv_to_df(csv_file):
    df = pd.read_csv(csv_file)

# Pull in all of the results csv files
csv_files = ["results.csv", "results2.csv", "results4.csv", "avg_results.csv", "avg_results2.csv", "avg_results3.csv", "avg_results4.csv", "vote_results.csv", "vote_results2.csv"]
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

df_max = df['acc'].max()
print("Max accuracy for df: ", df['acc'].max(), " at beta: ", df[df['acc'] == df_max]['beta'].values[0], " gamma: ", df[df['acc'] == df_max]['gamma'].values[0], " T: ", df[df['acc'] == df_max]['T'].values[0])
avg_df_max = avg_df['acc'].max()
print("Max accuracy for avg_df: ", avg_df['acc'].max(), " at beta: ", avg_df[avg_df['acc'] == avg_df_max]['beta'].values[0], " gamma: ", avg_df[avg_df['acc'] == avg_df_max]['gamma'].values[0], " T: ", avg_df[avg_df['acc'] == avg_df_max]['T'].values[0])
vote_df_max = vote_df['acc'].max()
print("Max accuracy for vote_df: ", vote_df['acc'].max(), " at beta: ", vote_df[vote_df['acc'] == vote_df_max]['beta'].values[0], " gamma: ", vote_df[vote_df['acc'] == vote_df_max]['gamma'].values[0], " T: ", vote_df[vote_df['acc'] == vote_df_max]['T'].values[0])

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

# plot_results(df) # Best beta ~ 2, 2.2 gamma ~ 0.6, 0.8 T ~ 100, 2 # T = 200, gamma = 0.6, beta before 1.9
# plot_results(avg_df) # Best beta ~ 2, gamma ~ 0.6, 0.8 T ~ 100, 2
# plot_results(vote_df) # Best beta ~ 2, gamma ~ 0.6 T ~ 2 insufficient data

def get_matching_T(df):
    # Get the set of unique t values
    t_values = df['T'].unique()

    # For each t, get the set of (a, b, c) tuples
    abc_sets = [set(tuple(row) for row in df[df['T'] == t][['beta', 'gamma']].values) for t in t_values]

    # Find intersection of all sets
    common_abc = set.intersection(*abc_sets)

    return common_abc

# beta_gamma = get_matching_T(avg_df)
# for pair in beta_gamma:
#     # Get the highest accuracy for each pair
#     df_pair = avg_df[(avg_df['beta'] == pair[0]) & (avg_df['gamma'] == pair[1])]
#     max_acc = df_pair['acc'].max()
#     print(f"Max accuracy for beta = {pair[0]}, gamma = {pair[1]}: {max_acc}")

# beta_best = 2.0
# gamma_best = 0.6
# df_best = avg_df[(avg_df['beta'] == beta_best) & (avg_df['gamma'] == gamma_best)]
# df_best = df_best.sort_values('T')
# T_vals = df_best['T']
# acc_vals = df_best['acc']
# plt.figure()
# plt.plot(T_vals, acc_vals, marker='o')
# plt.xlabel('T')
# plt.ylabel('Accuracy')
# plt.title(f'Accuracy vs T for beta = {beta_best}, gamma = {gamma_best}')
# plt.xticks(T_vals)
# plt.grid(True)
# plt.show()

def plot_results_T_as_x(df):
    # Get all the unique values of beta, gamma, and T
    betas = df['beta'].unique()
    gammas = df['gamma'].unique()
    Ts = df['T'].unique()

    # Convert negative accuracies to 0
    df['acc'] = df['acc'].apply(lambda x: max(x, 0))

    # Make hyperparameter plots
    for beta_val in betas:
        plt.figure()
        df_beta0 = df[df['beta'] == beta_val]
        for gamma in gammas:
            df_gamma = df_beta0[df_beta0['gamma'] == gamma]
            # print("df_gamma: ", df_gamma)
            if df_gamma.empty:
                continue
            plt.plot(df_gamma['T'], df_gamma['acc'], label=gamma)
        plt.xlabel('T')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs T for beta = ' + str(beta_val))
        plt.legend(title='Gamma')
    plt.show()

# plot_results_T_as_x(df) # Best beta ~ 2, gamma ~ 0.6, 0.8 T ~ 100, 2 # T = 200, gamma = 0.6, beta before 1.9
# plot_results_T_as_x(avg_df) # Best beta ~ 2, gamma ~ 0.6, 0.8 T ~ 100, 2
plot_results_T_as_x(vote_df) # Best beta ~ 2, gamma ~ 0.6 T ~ 2 insufficient data