"""
Plotting helpers used by the application notebook (Section 5).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def set_integer_xticks(ax, T, step=5):
    ticks = np.arange(0, T + 1, step)
    ax.set_xticks(ticks)
    ax.set_xlim(0, T)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))


def plot_irf_panel(ax, VAR, VAR_se, LP, LP_se, title, caption, xtick_step=5):
    h_var = np.arange(len(VAR))
    h_lp = np.arange(len(LP))
    ax.plot(h_var, VAR, label="VAR", color='navy')
    ax.fill_between(h_var, VAR - VAR_se, VAR + VAR_se, color='navy', alpha=0.2)
    ax.plot(h_lp, LP, label="LP", color='dimgray')
    ax.fill_between(h_lp, LP - LP_se, LP + LP_se, color='dimgray', alpha=0.2)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Estimator Value")
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    set_integer_xticks(ax, T=max(len(h_var), len(h_lp)) - 1, step=xtick_step)
    ax.text(0.50, -0.22, caption, transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic', clip_on=False)



def plot_application_hausman(applications, app_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=False)
    axes = axes.ravel()
    for ax, (key, app) in zip(axes, applications.items()):
        met = app_metrics[key]
        h = met["h"]
        ax.plot(h, met["abs_Delta"], marker='o', markersize=3, linewidth=1.8, color='navy')
        ax.set_title(app["title"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel('Hausman statistic')
        ax.set_ylim(0, 2.2)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        set_integer_xticks(ax, T=len(h)-1, step=app["xtick_step"])
        ax.text(0.50, -0.24, app["caption"], transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic', clip_on=False)
    plt.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.35)
    plt.show()


def plot_application_bias_correction(applications, app_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=False)
    axes = axes.ravel()

    for ax, (key, app) in zip(axes, applications.items()):
        met = app_metrics[key]

        h = np.asarray(met["h"])
        h1 = h >= 1
        bias_correction = np.asarray(met["bias_correction"])

        ax.plot(h[h1], bias_correction[h1], marker='o', markersize=3, linewidth=1.8, color='teal')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_title(app["title"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Bias correction")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        set_integer_xticks(ax, T=len(h)-1, step=app["xtick_step"])
        ax.set_xlim(left=1)

        ax.text(0.50, -0.24, app["caption"], transform=ax.transAxes, ha='center',
            va='top',fontsize=9, style='italic', clip_on=False)
    plt.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.35)
    plt.show()


def plot_application_corrected_irfs(applications, app_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=False)
    axes = axes.ravel()
    for ax, (key, app) in zip(axes, applications.items()):
        met = app_metrics[key]
        h = np.asarray(met["h"])
        h1 = h >= 1
        m = met["m"]
        omega_star = m**2 /(1 + m**2)

        VAR = np.asarray(met["VAR"])
        LP = np.asarray(met["LP"])
        MM = np.asarray(met["MM"])
        VAR_se = np.asarray(met["VAR_se"])
        LP_se = np.asarray(met["LP_se"])
        linear_opt = omega_star * LP + (1 - omega_star) * VAR

        ax.plot(h, VAR, label="VAR", color='navy', linewidth=1.8)
        ax.fill_between(h, VAR - VAR_se, VAR + VAR_se, color='navy', alpha=0.18)
        ax.plot(h, LP, label="LP", color='dimgray', linewidth=1.8)
        ax.fill_between(h, LP - LP_se, LP + LP_se, color='dimgray', alpha=0.18)
        ax.plot(h[h1], linear_opt[h1], label="Linear", color='darkorange', linestyle='--', linewidth=1.8)
        ax.plot(h[h1], MM[h1], label="Bias-corrected VAR", color='crimson', linestyle='-', linewidth=2)
        ax.set_title(app["title"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Estimator value")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        set_integer_xticks(ax, T=len(h)-1, step=app["xtick_step"])
        ax.text(0.50, -0.24, app["caption"], transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic', clip_on=False)
    handles = [
        Line2D([0], [0], color='navy', label='VAR'),
        Line2D([0], [0], color='dimgray', label='LP'),
        Line2D([0], [0], color='darkorange', linestyle='--', label='Optimal linear'),
        Line2D([0], [0], color='crimson', linewidth=2, label='Bias-corrected VAR')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=True, bbox_to_anchor=(0.5, 0.05), borderaxespad=0.2)
    plt.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.35)
    plt.show()


def plot_application_risk_improvement(applications, app_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=False)
    axes = axes.ravel()
    for ax, (key, app) in zip(axes, applications.items()):
        met = app_metrics[key]
        h = np.asarray(met["h"])
        h1 = h >= 1

        risk_improvement_pct = np.asarray(met["risk_improvement_pct"])

        ax.plot(h[h1], risk_improvement_pct[h1],marker='o',markersize=3,linewidth=1.8, color='crimson')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_title(app["title"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Risk improvement over linear (%)")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        set_integer_xticks(ax, T=len(h)-1, step=app["xtick_step"])
        ax.set_xlim(left=1)
        ax.text(0.50, -0.24, app["caption"], transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic', clip_on=False)
    plt.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.35)
    plt.show()
