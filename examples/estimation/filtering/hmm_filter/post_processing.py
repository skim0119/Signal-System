from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from jaxtyping import Array


def plot_filtering(
    times: Array,
    states: Array,
    observations: Array,
    beliefs: Array,
    save_path: Path | None = None,
    batch_index: int = 0,
) -> None:
    """Plot true state, observation, and filter beliefs for one batch item."""
    time = times
    state = states[:, batch_index].reshape(-1)
    observation = observations[:, batch_index].reshape(-1)
    belief = beliefs[:, batch_index, :]
    state_dim = belief.shape[-1]

    figure, axes = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )

    axes[0].step(time, state, where="post", color="C0", linewidth=1.5)
    axes[0].set_ylabel("State")
    axes[0].set_yticks(range(state_dim))
    axes[0].set_title("True hidden state")
    axes[0].grid(alpha=0.3)

    axes[1].step(time, observation, where="post", color="C1", linewidth=1.5)
    axes[1].set_ylabel("Observation")
    axes[1].set_title("Observation")
    axes[1].grid(alpha=0.3)

    for state_index in range(state_dim):
        axes[2].plot(
            time,
            belief[:, state_index],
            linewidth=1.5,
            label=f"P(state={state_index})",
        )
    axes[2].set_ylabel("Belief")
    axes[2].set_xlabel("Time")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Filter belief")
    axes[2].legend(loc="upper right", fontsize="small")
    axes[2].grid(alpha=0.3)

    figure.suptitle(f"HMM filtering (batch {batch_index})")

    if save_path is not None:
        figure.savefig(save_path, dpi=160)
        plt.close(figure)
        plt.close("all")
    else:
        plt.show()
