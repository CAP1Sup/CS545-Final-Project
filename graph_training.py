import os
import matplotlib.pyplot as plt
import sys


def plot_combined_results(folder_path: str):
    train_log_path = os.path.join("./results", folder_path, "log_train.txt")
    eval_log_path = os.path.join("./results", folder_path, "log_eval.txt")

    # Check if the log files exist
    missing_file = False
    if not os.path.exists(train_log_path):
        print(f"Training log file not found in {folder_path}")
        missing_file = True

    if not os.path.exists(eval_log_path):
        print(f"Evaluation log file not found in {folder_path}")
        missing_file = True

    if missing_file:
        return

    # Extract data from log_train.txt
    train_epochs = []
    train_losses = []
    with open(train_log_path, "r") as train_file:
        for line in train_file:
            if "Max iterations" in line:
                parts = line.split()
                max_iterations = int(parts[-1])
                continue

            if "Epoch" in line:
                parts = line.split()
                epoch = int(parts[1].strip(","))
                iter = int(parts[3].strip(","))
                mse_loss = float(parts[-1])

                # Pro-rate the epoch number to the max iterations
                # This is a simple linear scaling based on the number of iterations
                epoch += iter / max_iterations

                train_epochs.append(epoch)
                train_losses.append(mse_loss)

    # Extract data from log_eval.txt
    eval_epochs = []
    eval_mse_losses = []
    eval_restart_epochs = []
    with open(eval_log_path, "r") as eval_file:
        for line in eval_file:
            if "epoch" in line:
                # Skip the header line
                continue

            if "**Training restarted**" in line:
                # Note that the training was restarted
                eval_restart_epochs.append(eval_epochs[-1] if eval_epochs else 0)
                continue

            # The line contains evaluation data
            parts = line.split()
            epoch = int(parts[0].strip(":")) + 1  # At the end of the epoch
            mse_loss = float(parts[1].strip(","))
            eval_epochs.append(epoch)
            eval_mse_losses.append(mse_loss)

    # Plot the results
    plt.figure(figsize=(8, 6))

    # Plot training loss
    plt.plot(train_epochs, train_losses, label="Training Loss", color="blue")

    # Plot evaluation loss
    plt.plot(eval_epochs, eval_mse_losses, label="Evaluation Loss", color="orange")

    # Plot vertical lines for restarts
    for i, restart_epoch in enumerate(eval_restart_epochs):
        plt.axvline(
            x=restart_epoch,
            color="red",
            linestyle="--",
            label="Training Restarted" if i == 0 else None,
        )

    plt.xlabel("Epoch", fontsize=14)
    plt.xlim(0, max(train_epochs))
    plt.ylabel("Loss", fontsize=14)
    plt.title(
        f"Training and Evaluation Loss Over Epochs for {folder_path.strip('/')}",
        fontsize=13,
    )
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the results folder
    plot_path = os.path.join("./results", folder_path, "training_plot.svg")
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".svg", ".png"))

    # Display the plot
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python graph_training.py <model_architecture>")
        print("Example: python graph_training.py timm/tinynet_a")
        sys.exit(1)

    model = str(sys.argv[1])
    plot_combined_results(model)
