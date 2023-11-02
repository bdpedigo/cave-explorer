def clean_axis(ax):
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
