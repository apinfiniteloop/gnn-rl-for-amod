import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the data
log = torch.load(r"ltm_lr1e-3_beta1.pth")

# Extract the data and convert to numpy arrays
train_reward = np.array(log["train_reward"])
train_served_demand = np.array(log["train_served_demand"])
train_reb_cost = np.array(log["train_reb_cost"])


# Define a function to calculate rolling average
def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Calculate rolling averages with a window size of 10
window_size = 200
rolling_reward = rolling_average(train_reward, window_size)
rolling_served_demand = rolling_average(train_served_demand, window_size)
rolling_reb_cost = rolling_average(train_reb_cost, window_size)

# Plot original data
plt.plot(train_reward, label="Train Reward", color="blue", alpha=0.25)
plt.plot(train_served_demand, label="Train Served Demand", color="green", alpha=0.25)
plt.plot(train_reb_cost, label="Train Reb Cost", color="red", alpha=0.25)

# Plot rolling averages with different line styles
plt.plot(
    np.arange(window_size - 1, len(train_reward)),
    rolling_reward,
    linestyle="--",
    color="blue",
    label="Rolling Avg Reward",
)
plt.plot(
    np.arange(window_size - 1, len(train_served_demand)),
    rolling_served_demand,
    linestyle="--",
    color="green",
    label="Rolling Avg Served Demand",
)
plt.plot(
    np.arange(window_size - 1, len(train_reb_cost)),
    rolling_reb_cost,
    linestyle="--",
    color="red",
    label="Rolling Avg Reb Cost",
)

# Add legend and labels
plt.legend()
plt.title("Training Metrics with Rolling Averages")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.show()
