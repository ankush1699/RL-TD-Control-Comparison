"""
Plotting helpers. Keeps figure styling consistent across notebooks.
"""
import matplotlib.pyplot as plt


# colors for the four methods, consistent everywhere
METHOD_COLORS = {
    "SARSA": "#1f77b4",
    "Q-learning": "#ff7f0e",
    "Expected SARSA": "#2ca02c",
    "RR-ES": "#d62728",
}


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)