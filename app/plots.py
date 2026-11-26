"""Matplotlib figure for the dashboard day view, styled for the dark theme."""
import matplotlib.pyplot as plt
import numpy as np

BG, PANEL, TEXT = "#0e1a19", "#0e1a19", "#e6efed"
TEAL, AMBER, WHITE = "#2dd4bf", "#fbbf24", "#f4f7f6"


def day_figure(q, actual, bench, buy, cap=10.0):
    """q: (96, 3) P10/P50/P90; actual, bench: (96,); buy: (96,) bool."""
    hours = np.arange(96) / 4
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.fill_between(hours, q[:, 0], q[:, 2], color=TEAL, alpha=0.20, label="P10-P90 band")
    ax.plot(hours, q[:, 1], color=TEAL, lw=2.0, label="P50 forecast")
    ax.plot(hours, actual, color=WHITE, lw=1.2, label="actual price")
    ax.plot(hours, bench, color=AMBER, lw=1.3, ls="--", label="benchmark (7-day block mean)")
    ax.scatter(hours[buy], np.full(int(buy.sum()), 0.2), marker="s", s=16, color=TEAL,
               label="agent: buy on exchange")
    ax.axhline(cap, color=TEXT, lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, cap + 0.8)
    ax.set_xlabel("hour of day", color=TEXT, fontsize=12)
    ax.set_ylabel("Rs/kWh", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT, labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=10, frameon=False,
              labelcolor=TEXT)
    fig.tight_layout()
    return fig
