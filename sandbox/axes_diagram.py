# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Use matplotlib hand drawn theme
plt.xkcd()
sns.set_context("talk")

# Create a figure and axes
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

# Set the labels
ax.set_xlabel("Proofreading")
ax.set_ylabel("Connectivity feature")

ax.spines[["top", "right"]].set_visible(False)
