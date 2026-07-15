import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt 

from train import full_train_test_val, parse_args


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "results" / "losses.png"


def plot_losses(train_losses, val_losses, test_loss=None, output_path=DEFAULT_OUTPUT_PATH):
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length")
    if not train_losses:
        raise ValueError("at least one loss value is required")

    epochs = list(range(1, len(train_losses) + 1))
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs, val_losses, marker="o", label="Validation loss")
    if test_loss is not None:
        ax.hlines(
            test_loss,
            xmin=epochs[0],
            xmax=epochs[-1],
            colors="tab:green",
            linestyles="--",
            label=f"Test loss ({test_loss:.4f})",
        )

    ax.set_title("Fashion-MNIST MLP losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv=None, output_path=DEFAULT_OUTPUT_PATH):
    args = parse_args(argv)
    train_losses, val_losses, test_loss = full_train_test_val(args)
    saved_path = plot_losses(train_losses, val_losses, test_loss, output_path)
    print(f"saved_loss_plot={saved_path}")
    return saved_path
    

if __name__ == "__main__":
    main()
