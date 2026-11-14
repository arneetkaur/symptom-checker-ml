import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("robustness_results.csv")
styles = {"Logistic Regression": ("o", "-"),
          "Random Forest": ("s", "--"),
          "SVM": ("^", ":")}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (key, title) in zip(axes, [("top1", "Top-1 accuracy"),
                                   ("top3", "Top-3 accuracy")]):
    for name, (marker, line) in styles.items():
        d = df[df.model == name].sort_values("symptoms_kept")
        ax.errorbar(d.symptoms_kept * 100, d[key + "_mean"],
                    yerr=d[key + "_std"], marker=marker, linestyle=line,
                    capsize=3, label=name)
    ax.set_title(title)
    ax.set_xlabel("Reported symptoms kept (%)")
    ax.set_xticks([25, 50, 75, 100])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Accuracy (mean ± std, 20 trials)")
axes[0].legend(loc="lower right")
plt.tight_layout()
plt.savefig("robustness_figure.png", dpi=200)