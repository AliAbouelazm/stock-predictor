"""Retro pixel-style theme for visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PIXEL_COLORS = {
    "up": "#00FF00",
    "down": "#FF0000",
    "flat": "#FFFF00",
    "price": "#0000FF",
    "background": "#000000",
    "grid": "#333333",
    "text": "#FFFFFF"
}


def apply_pixel_style(fig, ax):
    """Apply retro pixel-style theme to matplotlib figure."""
    ax.set_facecolor(PIXEL_COLORS["background"])
    fig.patch.set_facecolor(PIXEL_COLORS["background"])
    
    ax.grid(True, color=PIXEL_COLORS["grid"], linewidth=2, linestyle="-", alpha=0.5)
    ax.spines["top"].set_color(PIXEL_COLORS["grid"])
    ax.spines["bottom"].set_color(PIXEL_COLORS["grid"])
    ax.spines["left"].set_color(PIXEL_COLORS["grid"])
    ax.spines["right"].set_color(PIXEL_COLORS["grid"])
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.tick_params(colors=PIXEL_COLORS["text"], labelsize=10, width=2)
    ax.xaxis.label.set_color(PIXEL_COLORS["text"])
    ax.yaxis.label.set_color(PIXEL_COLORS["text"])
    ax.title.set_color(PIXEL_COLORS["text"])
    
    return fig, ax


def plot_pixel_line(ax, x, y, color, label=None, linewidth=3):
    """Plot line with pixel-style appearance."""
    ax.plot(x, y, color=color, linewidth=linewidth, marker="s", markersize=4, 
            markevery=max(1, len(x)//50), label=label, drawstyle="steps-mid")


def plot_pixel_bar(ax, x, y, color, width=0.8):
    """Plot bar chart with pixel-style appearance."""
    ax.bar(x, y, color=color, width=width, edgecolor=PIXEL_COLORS["text"], linewidth=2)

