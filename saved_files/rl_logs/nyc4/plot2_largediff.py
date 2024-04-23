import torch
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib to use LaTeX for text rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def plot_and_save(data_path):
    # Load the data
    data = torch.load(data_path)

    # Define the keys to plot
    keys = ["train_reward", "train_reb_cost", "train_served_demand"]

    legend = ["Reward", "Rebalancing Cost", "Served Demand"]

    # Create a figure
    fig, ax = plt.subplots()

    # Colors for each key
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Process each data series
    for key, color, txt in zip(keys, colors, legend):
        # Original data
        original_data = data[key]

        # Smooth out sudden jumps greater than 10000
        for i in range(1, len(original_data)):
            if abs(original_data[i] - original_data[i - 1]) > 10000:
                original_data[i] = original_data[i - 1]

        ax.plot(original_data, alpha=0.25, color=color)

        # Rolling average data
        rolling_avg = (
            pd.Series(original_data)
            .rolling(window=min(500, len(original_data)), min_periods=1)
            .mean()
        )
        ax.plot(rolling_avg, label=txt, alpha=1.0, color=color)

    # Set the title and labels
    # ax.set_title("Training Metrics with Rolling Averages")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Values")

    # Add a legend
    ax.legend()

    # Use tight layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig("training_metrics_ldiff.pdf")


# Assuming 'example_data.pth' is the file path
plot_and_save("a2c_gnn_test.pth")
