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
            plt.xlabel('k values')
            plt.ylabel(metric_name)
            # plt.title(f'{metric_name} vs k for Different Models')
            plt.legend(title='Model')
            plt.grid(True)

            # Show plot
            plt.show()


if __name__ == '__main__':
    plot()
