import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    (88.54, 15.68, 0.7955726225434832, 1.0),
    (118.44, 11.12, 1.0, 1.0),
    (91.08, 15.0, 0.6710584101888449, 1.0),
    (92.64, 11.8, 1.0, 1.0),
    (97.82, 17.66, 1.0, 1.0),
    (71.54, 10.36, 1.0, 1.0)
]

# Unpacking data
mean_cycles_35t, mean_cycles_4, cycle_keep_rate_35t, cycle_keep_rate_4 = zip(*data)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot 1: Distribution of mean cycles
ax1.scatter(np.zeros_like(mean_cycles_35t), mean_cycles_35t, c='blue', alpha=0.6, s=100, label='3.5t')
ax1.scatter(np.ones_like(mean_cycles_4), mean_cycles_4, c='red', alpha=0.6, s=100, label='4')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['GPT-3.5-turbo', 'GPT-4'])
ax1.set_title('Mean Cycles Per Prompt')
ax1.set_ylabel('Mean Cycles')
ax1.set_ylim(0, 130)
ax1.set_xlim(-0.5, 1.5)

# Plot 2: Distribution of cycle keep rates
ax2.scatter(np.zeros_like(cycle_keep_rate_35t), cycle_keep_rate_35t, c='blue', alpha=0.6, s=100, label='3.5t')
ax2.scatter(np.ones_like(cycle_keep_rate_4), cycle_keep_rate_4, c='red', alpha=0.6, s=100, label='4')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['GPT-3.5-turbo', 'GPT-4'])
ax2.set_title('Cycle Retention Per Prompt')
ax2.set_ylabel('Cycle Retention')
ax2.set_ylim(0, 1)
ax2.set_xlim(-0.5, 1.5)


# Adjust layout and show plot
plt.tight_layout()
plt.show()