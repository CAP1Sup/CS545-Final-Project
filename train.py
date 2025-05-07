import os
import sys
import argparse
import random

sys.path.append(os.getcwd())

import torch
import numpy as np
from tqdm import tqdm
import pickle

from src.arcface import build_arcface_model
from src.crop import crop_and_resize
from src.student import StudentNetwork

STYLEGAN_PATH = "./stylegan2-ffhq-256x256.pkl"

# Set the random seeds for reproducibility
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(
    description="Script to train the student network using SynthDistill method"
)
parser.add_argument(
    "--model",
    metavar="<model>",
    type=str,
    default="timm/tinynet_a",
    help="Model name to train. Model names are in HuggingFace using format: author/model_name",
)
parser.add_argument(
    "--resampling_coef",
    metavar="<resampling_coef>",
    type=float,
    default=1.0,
    help="resampling coefficient",
)
parser.add_argument(
    "--gpu_id",
    metavar="<gpu_id>",
    type=int,
    default=0,
    help="GPU ID to use",
)
parser.add_argument(
    "--batch_size",
    metavar="<batch_size>",
    type=int,
    default=64,
    help="Batch size for training",
)
parser.add_argument(
    "--epochs",
    metavar="<epochs>",
    type=int,
    default=25,
    help="Number of epochs to train",
)

# Process the command line arguments
args = parser.parse_args()
resampling_coef = args.resampling_coef

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
print("The device being used for training is:", device)

batch_size = args.batch_size
start_epoch = 0
num_epochs = args.epochs
iterations_per_epoch_train = int(1e6 / batch_size)
iterations_per_epoch_val = int(1e4 / batch_size)

# Save the losses over time
# This is used to stop the training if the loss is increasing for 3 epochs in a row
losses = []

# Set up the results folder
# If the model's folder exists, resume training
results_path = f"results/{args.model}/"
previous_training = False
try:
    os.makedirs(results_path)
    with open(results_path + "/log_train.txt", "w") as f:
        f.write(f"Max iterations: {iterations_per_epoch_train}\n")
    with open(results_path + "/log_eval.txt", "w") as f:
        f.write("epoch, loss_eval_MSE, loss_eval_cos\n")
except OSError:
    print("Model folder already exists in results, resuming training")
    print("If you want to start a new training, delete the model folder")
    previous_training = True
    with open(results_path + "/log_eval.txt", "a") as f:
        f.write("**Training restarted**\n")

# Determine the model to use
student = StudentNetwork(model_name=args.model)
print("Model params: ", sum(p.numel() for p in student.parameters()))

if previous_training:
    # Get all of the files in the results folder, then filter for the .pt files
    # Sort the files by the epoch number
    files = os.listdir(results_path)
    files = [f for f in files if f.endswith(".pt")]
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Load the last checkpoint
    student.load_state_dict(torch.load(results_path + files[-1]))

    # Get the epoch number from the last checkpoint
    start_epoch = int(files[-1].split("_")[-1].split(".")[0]) + 1
    print(f"Resuming training from epoch {start_epoch}")

student.to(device)


teacher = build_arcface_model(device)

# Load the StyleGAN model
sys.path.append("./stylegan3")
with open(STYLEGAN_PATH, "rb") as f:
    StyleGAN = pickle.load(f)["G_ema"]

    # Load the synthesis and mapping networks
    StyleGAN_synthesis = StyleGAN.synthesis
    StyleGAN_mapping = StyleGAN.mapping

    # Set the networks to evaluation mode
    StyleGAN_synthesis.eval()
    StyleGAN_mapping.eval()

    # Move the networks to the specified device
    StyleGAN_synthesis.to(device)
    StyleGAN_mapping.to(device)

z_dim_StyleGAN = StyleGAN.z_dim

# Setup the loss function for evaluating between the two networks
MSELoss = torch.nn.MSELoss(reduction="mean")

# Learning rate and optimizer
lr = 0.001
optimizer = torch.optim.Adam(student.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# Convenience function to rescale the images
def process_imgs(imgs):
    # Clamp image values to prevent issues
    imgs = torch.clamp(imgs, min=-1, max=1)

    # Rescale image values to the range (0, 1)
    imgs = (imgs + 1) / 2.0

    # Crop and resize the images to the desired size
    imgs = crop_and_resize(imgs)

    return imgs


# Training loop
for epoch in range(start_epoch, num_epochs):
    student.train()

    for itr in tqdm(range(iterations_per_epoch_train), desc="Training"):
        with torch.no_grad():
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, z_dim_StyleGAN, device=device)

            # Generate w from noise(z)
            w = StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()

            # Synthesize images from w
            imgs = process_imgs(StyleGAN_synthesis(w).detach())

        # Initial forward pass
        student_emb = student((imgs - 0.5) / 0.5)  # Scale to [-1, 1]
        with torch.no_grad():
            teacher_emb = teacher.transform(imgs)
        MSE = MSELoss(student_emb, teacher_emb)

        # Initial backward pass
        optimizer.zero_grad()
        MSE.backward()
        optimizer.step()

        # Generate a new batch of latent vectors from the worst similarity
        # between the student and teacher embeddings
        # This will focus the training on the most difficult samples
        cos_sim = torch.nn.CosineSimilarity()(student_emb, teacher_emb)
        coef = (cos_sim + 1) / 2.0  # Normalize to [0, 1]
        w_ = w[:, 0, :]
        w_ = w_ + coef.unsqueeze(1) * resampling_coef * torch.randn(
            batch_size, StyleGAN.w_dim, device=device
        )
        w = w_.unsqueeze(1).repeat([1, StyleGAN.num_ws, 1])

        with torch.no_grad():
            # Synthesize images from w
            imgs = process_imgs(StyleGAN_synthesis(w).detach())

        # Focused forward pass
        student_emb = student((imgs - 0.5) / 0.5)  # Scale to [-1, 1]
        with torch.no_grad():
            teacher_emb = teacher.transform(imgs)

        MSE = MSELoss(student_emb, teacher_emb)
        cos = torch.nn.CosineSimilarity()(student_emb, teacher_emb).mean()

        # Focused backward pass
        optimizer.zero_grad()
        MSE.backward()
        optimizer.step()

        # Print the loss every 500 iterations
        if itr % 500 == 0:
            output = f"Epoch: {epoch}, Iteration: {itr}, Loss: {MSE.item()}"

            print(output, flush=True)

            with open(results_path + "/log_train.txt", "a") as f:
                f.write(output + "\n")

    # Set the model to evaluation mode (should not be learning on the validation set)
    student.eval()

    # Save the model
    torch.save(student.state_dict(), results_path + "/epoch_" + str(epoch) + ".pt")

    # Evaluate the model on the validation set
    loss_eval_MSE = loss_eval_cos = 0
    for itr in tqdm(range(iterations_per_epoch_val), desc="Validation"):
        with torch.no_grad():
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, z_dim_StyleGAN, device=device)

            # Generate w from noise(z)
            w = StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()

            # Synthesize images from w
            imgs = process_imgs(StyleGAN_synthesis(w).detach())

            # Only forward pass
            student_emb = student((imgs - 0.5) / 0.5)  # Scale to [-1, 1]
            teacher_emb = teacher.transform(imgs)
            MSE = MSELoss(student_emb, teacher_emb)
            cos = torch.nn.CosineSimilarity()(student_emb, teacher_emb).mean()

            # Accumulate the losses
            loss_eval_MSE += MSE.item()
            loss_eval_cos += cos.item()

    with open(results_path + "/log_eval.txt", "a") as f:
        f.write(
            f"{epoch}: {loss_eval_MSE / iterations_per_epoch_val}, {loss_eval_cos / iterations_per_epoch_val}\n"
        )

    # Save the loss
    losses.append(loss_eval_MSE / iterations_per_epoch_val)

    # Check if the loss is increasing for the last 3 epochs
    if len(losses) > 3 and all(losses[-i] > losses[-(i + 1)] for i in range(1, 4)):
        print("Loss is increasing, stopping training")
        break

    scheduler.step()
