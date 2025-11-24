import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Output dir (Desktop/changyidan)
SAVE_DIR = os.path.expanduser("~/Desktop/changyidan")
os.makedirs(SAVE_DIR, exist_ok=True)

# Utilities (rounded box + arrow with label)
def draw_box(ax, center, w, h, text, fontsize=18, z=3):
    x, y = center
    x0, y0 = x - w/2, y - h/2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.6, edgecolor="black", facecolor="white", zorder=z
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=z+1)

def draw_arrow(ax, start, end, text=None, text_offset=(0,0), rad=0.0, z=2, lw=2.6):
    """rad: curvature (0 straight; ±0.12 mild curve)"""
    arrow = FancyArrowPatch(
        start, end, mutation_scale=14, arrowstyle="->",
        connectionstyle=f"arc3,rad={rad}", linewidth=lw, color="black", zorder=z
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=16, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9), zorder=z+1)

# (1) LSTM gradient highway (fixed margins/labels)
def fig_lstm_highway(save_path):
    fig, ax = plt.subplots(figsize=(12.5, 5.4), dpi=180)
    ax.set_xlim(-0.25, 6.65)
    ax.set_ylim(-0.10, 3.15)
    ax.axis("off")

    ax.set_title(r"LSTM gradient highway: when $f_t \approx 1$, $\partial c_t/\partial c_{t-1} \approx 1$",
                 fontsize=22, pad=16)

    pos_c_prev   = (0.8, 2.1)
    pos_c        = (5.0, 2.1)
    pos_xh       = (0.8, 0.7)
    pos_c_tilde  = (2.6, 0.7)
    pos_h        = (5.0, 0.7)

    draw_box(ax, pos_c_prev, 1.4, 0.72, r"$c_{t-1}$")
    draw_box(ax, pos_c,      1.4, 0.72, r"$c_{t}$")
    draw_box(ax, pos_xh,     1.6, 0.72, r"$x_t,\;h_{t-1}$")
    draw_box(ax, pos_c_tilde,1.6, 0.72, r"$\tilde{c}_{t}$")
    draw_box(ax, pos_h,      1.4, 0.72, r"$h_{t}$")

    # c_{t-1} -> c_t
    draw_arrow(ax,
               (pos_c_prev[0]+0.7, pos_c_prev[1]),
               (pos_c[0]-0.7,     pos_c[1]),
               text=r"$f_t$ (forget gate)",
               text_offset=(0.00, 0.28),
               rad=0.0)

    # \tilde{c}_t -> c_t
    draw_arrow(ax,
               (pos_c_tilde[0]+0.80, pos_c_tilde[1]+0.22),
               (pos_c[0]-0.72,       pos_c[1]-0.16),
               text=r"$i_t \odot \tilde{c}_t$",
               text_offset=(-0.08, 0.18),
               rad=0.0)

    # (x_t,h_{t-1}) -> \tilde{c}_t
    draw_arrow(ax,
               (pos_xh[0]+0.80, pos_xh[1]),
               (pos_c_tilde[0]-0.80, pos_c_tilde[1]),
               text="nonlinear",
               text_offset=(-0.02, 0.26),
               rad=0.10)

    start_v = (pos_c[0]+0.00, pos_c[1]-0.36)
    end_v   = (pos_h[0]+0.00, pos_h[1]+0.36)
    draw_arrow(ax, start_v, end_v)
    ax.text(start_v[0]-0.75, (start_v[1]+end_v[1])/2, r"$\tanh$",
            fontsize=16, ha="right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ax.text(end_v[0]+0.62, (start_v[1]+end_v[1])/2, r"output gate $o_t$",
            fontsize=16, ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# (2) GRU
def fig_gru_flow(save_path):
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=180)
    ax.set_xlim(0, 6); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title(r"GRU update blends previous state and candidate via $z_t$",
                 fontsize=22, pad=16)

    pos_h_prev   = (0.9, 1.6)
    pos_gate_z   = (2.9, 2.6)
    pos_h_tilde  = (2.9, 0.7)
    pos_h        = (5.0, 1.6)
    pos_xh       = (0.9, 0.7)

    draw_box(ax, pos_h_prev, 1.6, 0.72, r"$h_{t-1}$")
    draw_box(ax, pos_gate_z, 2.0, 0.70, r"$z_t$ (update gate)", fontsize=17)
    draw_box(ax, pos_h_tilde,1.8, 0.72, r"$\tilde{h}_t$")
    draw_box(ax, pos_h,      1.6, 0.72, r"$h_t$")
    draw_box(ax, pos_xh,     1.6, 0.72, r"$x_t,\;h_{t-1}$")

    draw_arrow(ax, (pos_h_prev[0]+0.8, pos_h_prev[1]),
                    (pos_h[0]-0.8,     pos_h[1]),
                    text=r"$z_t \odot h_{t-1}$",
                    text_offset=(0,0.26), rad=0.06)

    draw_arrow(ax, (pos_h_tilde[0]+0.9, pos_h_tilde[1]+0.2),
                    (pos_h[0]-0.8,     pos_h[1]-0.2),
                    text=r"$(1-z_t)\odot \tilde{h}_t$",
                    text_offset=(0.02,0.22), rad=0.0)

    draw_arrow(ax, (pos_xh[0]+0.8, pos_xh[1]),
                    (pos_h_tilde[0]-0.9, pos_h_tilde[1]),
                    text=r"reset gate $r_t$",
                    text_offset=(0,0.26), rad=0.10)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.07)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# (3) Gradient norm plot
def fig_rnn_gradient(save_path, T=50):
    t = np.arange(T+1)
    g1 = 0.9 ** t
    g2 = 1.0 ** t
    g3 = 1.2 ** t

    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=180)
    ax.plot(t, g1, label="effective gain < 1 (vanishing)", linewidth=3)
    ax.plot(t, g2, label="effective gain ≈ 1 (stable-ish)", linewidth=3)
    ax.plot(t, g3, label="effective gain > 1 (exploding)", linewidth=3)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlabel("Time step", fontsize=16)
    ax.set_ylabel("Gradient norm (log scale)", fontsize=16)
    ax.set_title("Gradient Norm Over Time in a Simple RNN (toy demonstration)", fontsize=20, pad=12)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Run all
if __name__ == "__main__":
    fig_lstm_highway(os.path.join(SAVE_DIR, "lstm_gradient_highway.png"))
    fig_gru_flow(os.path.join(SAVE_DIR, "gru_gradient_flow.png"))
    fig_rnn_gradient(os.path.join(SAVE_DIR, "rnn_gradient_flow.png"))
    print("Saved to:", SAVE_DIR)
