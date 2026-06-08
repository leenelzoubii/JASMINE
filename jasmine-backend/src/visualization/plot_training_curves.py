import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(csv_path):
    """
    Reads per_epoch_metrics.csv and creates training vs validation
    accuracy and loss plots for each model.
    """

    df = pd.read_csv(csv_path)

    print("CSV columns:", df.columns.tolist())

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [
        "model",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model].copy()

        # If there are multiple folds, average the metrics per epoch
        model_df = (
            model_df
            .groupby("epoch", as_index=False)
            .agg({
                "train_loss": "mean",
                "val_loss": "mean",
                "train_acc": "mean",
                "val_acc": "mean"
            })
        )

        # -------------------------------
        # Accuracy Curve
        # -------------------------------
        plt.figure(figsize=(8, 5))

        plt.plot(
            model_df["epoch"],
            model_df["train_acc"],
            marker="o",
            label="Training Accuracy"
        )

        plt.plot(
            model_df["epoch"],
            model_df["val_acc"],
            marker="o",
            label="Validation Accuracy"
        )

        plt.title(f"{model.upper()} Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        accuracy_path = output_dir / f"{model}_accuracy_curve.png"
        plt.savefig(accuracy_path, dpi=300)
        plt.close()

        print(f"Saved accuracy plot: {accuracy_path}")

        # -------------------------------
        # Loss Curve
        # -------------------------------
        plt.figure(figsize=(8, 5))

        plt.plot(
            model_df["epoch"],
            model_df["train_loss"],
            marker="o",
            label="Training Loss"
        )

        plt.plot(
            model_df["epoch"],
            model_df["val_loss"],
            marker="o",
            label="Validation Loss"
        )

        plt.title(f"{model.upper()} Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        loss_path = output_dir / f"{model}_loss_curve.png"
        plt.savefig(loss_path, dpi=300)
        plt.close()

        print(f"Saved loss plot: {loss_path}")


if __name__ == "__main__":
    csv_file = Path("results/per_epoch_metrics.csv")
    plot_training_curves(csv_file)