import pandas as pd
from matplotlib import pyplot as plt


def plot():
    results = pd.read_csv('new_results.csv')

    print(results)

    grouped_df = results.groupby(['Dataset', 'Metric'], as_index=False)

    for name, group in grouped_df:
        print(f"Group: {name}")
        model_name, metric_name = name  # unpack the model and metric names
        if metric_name == 'F1':
            plt.rcParams.update({
                'font.size': 14,  # General font size
                'axes.titlesize': 16,  # Title font size
                'axes.labelsize': 16,  # X and Y axis label font size
                'xtick.labelsize': 14,  # X tick label font size
                'ytick.labelsize': 14,  # Y tick label font size
                'legend.fontsize': 14  # Legend font size
            })
            # Melt the DataFrame to make it long-form for plotting
            df_melted = group.melt(id_vars=['Model', 'Dataset', 'Metric', 'MAP'],
                                   value_vars=['k_1', 'k_5', 'k_10', 'k_15', 'k_20', 'k_25', 'k_30', 'k_35', 'k_40', 'k_45',
                                               'k_50', 'k_100'],
                                   var_name='k',
                                   value_name='Metric Value')

            # Convert 'k' column to numeric values
            df_melted['k'] = df_melted['k'].str.replace('k_', '').astype(int)

            # Plotting
            plt.figure(figsize=(10, 6))

            # Plot for each dataset (you can adjust the 'Dataset' and 'Metric' filters as needed)
            for dataset in df_melted['Model'].unique():
                subset = df_melted[df_melted['Model'] == dataset]
                plt.plot(subset['k'], subset['Metric Value'], marker='o', label=dataset)

            # Adding labels and title
            plt.xlabel('k values', fontsize=16)
            plt.ylabel('F score', fontsize=16)  # Assuming 'metric_name' is a variable containing the y-axis label
            plt.legend(title='Model', title_fontsize=14, fontsize=14)
            plt.grid(True)

            # Show plot
            plt.show()


if __name__ == '__main__':
    plot()
