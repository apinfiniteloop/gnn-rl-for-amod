import torch
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib to use LaTeX for text rendering
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def plot_and_save(data_path1, data_path2, data_path3):
    # Load the data from both files
    data1 = torch.load(data_path1)
    data2 = torch.load(data_path2)
    data3 = torch.load(data_path3)

    # Create a figure
    fig, ax = plt.subplots()

    # Colors for each dataset
    color1 = "#31708E"  # Muted teal
    color2 = "#843C39"  # Soft burgundy
    color3 = "#556B2F"  # Olive green

    # Plot data and rolling average for the first dataset
    train_reward1 = data1["train_reward"]
    ax.plot(train_reward1, alpha=0.25, color=color1)
    rolling_avg1 = (
        pd.Series(train_reward1)
        .rolling(window=min(500, len(train_reward1)), min_periods=1)
        .mean()
    )
    ax.plot(rolling_avg1, label=r"$\mu=0$(\texttt{NoGenCost})", alpha=1.0, color=color1)

    # Plot data and rolling average for the second dataset
    train_reward2 = data2["train_reward"]
    ax.plot(train_reward2, alpha=0.25, color=color2)
    rolling_avg2 = (
        pd.Series(train_reward2)
        .rolling(window=min(500, len(train_reward2)), min_periods=1)
        .mean()
    )
    ax.plot(rolling_avg2, label=r"$\mu=1$(\texttt{GenCost1})", alpha=1.0, color=color2)

    train_reward3 = data3["train_reward"]
    for i in range(1, len(train_reward3)):
        if abs(train_reward3[i] - train_reward3[i - 1]) > 10000:
            train_reward3[i] = train_reward3[i - 1]
    ax.plot(train_reward3, alpha=0.25, color=color3)
    rolling_avg3 = (
        pd.Series(train_reward3)
        .rolling(window=min(500, len(train_reward3)), min_periods=1)
        .mean()
    )
    ax.plot(rolling_avg3, label=r"$\mu=3$(\texttt{Taylor})", alpha=1.0, color=color3)

    # Set the title and labels
    # ax.set_title("Comparison of Train Rewards")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Train Reward")

    # Add a legend
    ax.legend()

    # Use tight layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig("comparison_train_rewards.pdf")


# Assuming 'data1.pth' and 'data2.pth' are the file paths
plot_and_save(
    "ltm_lr5e-4_beta1_notaylor.pth",
    "ltm_lr5e-4_beta1.pth",
    "ltm_lr5e-4_beta1_3taylor.pth",
)
