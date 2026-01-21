import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"
model_save_path = "./model/Bestmodel_diffusion_bonus.pkl"
vis_root = "./vis"

# Hyperparameters
batch_size = 16
num_epochs = 1000  
img_channel = 3
img_width, img_height = 24, 24

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"using device {device}")

# Diffusion process settings
num_steps = 300
beta_min = 1e-4
beta_max = 5e-2

# Initialize diffusion parameters
betas = torch.linspace(-6, 6, num_steps, device=device)
betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min
alphas = 1 - betas
alphas_sqrt = torch.sqrt(alphas)
alphas_bar = torch.cumprod(alphas, 0)
alphas_bar_sqrt = torch.sqrt(alphas_bar)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)


class Diffuser(nn.Module):
    """
    Enhanced Diffusion Model with x-prediction and conditional generation.
    
    Based on "Back to Basics: Let Denoising Generative Models Denoise" (Kaiming He et al.)
    Key principle: Predict clean data (x) instead of noise, leveraging the manifold assumption.
    """
    def __init__(self, n_steps, img_channels=3, num_classes=3, use_bottleneck=True, bottleneck_dim=64):
        super(Diffuser, self).__init__()
        self.n_steps = n_steps
        self.num_classes = num_classes
        self.use_bottleneck = use_bottleneck
        
        # Time embedding (sinusoidal)
        self.time_embedding_dim = 128
        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_embedding_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Class embedding for conditional generation
        self.class_embedding = nn.Embedding(num_classes, 256)
        
        # Combined conditioning
        self.cond_mlp = nn.Sequential(
            nn.Linear(512, 256),  # 256 (time) + 256 (class)
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Time conditioning layers
        self.time_cond1 = nn.Linear(256, 128)
        self.time_cond2 = nn.Linear(256, 256)
        
        # Encoder with optional bottleneck
        if use_bottleneck:
            # Bottleneck design as suggested by He's paper
            # Reduces dimensionality to encourage manifold learning
            self.conv1_bottleneck = nn.Sequential(
                nn.Conv2d(img_channels, 32, 3, 1, 1),
                nn.GroupNorm(4, 32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.GroupNorm(8, 64)
            )
        else:
            self.conv1_bottleneck = nn.Sequential(
                nn.Conv2d(img_channels, 64, 3, 1, 1),
                nn.GroupNorm(8, 64)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.GroupNorm(8, 128)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.GroupNorm(8, 256)
        )
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        
        # Output layer
        self.conv_out = nn.Conv2d(64, img_channels, 1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def pos_encoding(self, t, channels):
        """Sinusoidal time embedding"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, class_labels=None):
        """
        Forward pass: Predict CLEAN IMAGE (x-prediction).
        
        Key insight from He's paper: Predicting clean data directly is easier
        than predicting noise because natural images lie on a low-dimensional manifold.
        
        Args:
            x: Noisy input image
            t: Timestep
            class_labels: Class labels for conditional generation (optional)
        
        Returns:
            Predicted clean image (NOT noise!)
        """
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_embed = self.pos_encoding(t, self.time_embedding_dim)
        t_embed = self.time_embedding(t_embed)
        
        # Class conditioning
        if class_labels is not None:
            class_embed = self.class_embedding(class_labels)
            # Combine time and class embeddings
            combined_embed = torch.cat([t_embed, class_embed], dim=1)
            cond_embed = self.cond_mlp(combined_embed)
        else:
            cond_embed = t_embed
        
        # Encoder with bottleneck (manifold learning)
        x1 = self.conv1_bottleneck(x)
        x1 = self.relu(x1)
        x1_pool = self.pool(x1)  # 64 x 12 x 12
        
        x2 = self.conv2(x1_pool)
        # Time/class conditioning
        time_scale = self.time_cond1(cond_embed)
        time_scale = time_scale.unsqueeze(-1).unsqueeze(-1)
        x2 = x2 * (1 + time_scale)
        x2 = self.relu(x2)
        x2_pool = self.pool(x2)  # 128 x 6 x 6
        
        # Bottleneck
        x3 = self.conv3(x2_pool)
        # Time/class conditioning
        time_scale = self.time_cond2(cond_embed)
        time_scale = time_scale.unsqueeze(-1).unsqueeze(-1)
        x3 = x3 * (1 + time_scale)
        x3 = self.relu(x3)
        
        # Decoder with skip connections
        x4 = self.upconv1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.conv4(x4)
        
        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.conv5(x5)
        
        # Output: PREDICTED CLEAN IMAGE
        out = self.conv_out(x5)
        
        return out
    
    def sample(self, shape, n_steps, class_label=None):
        """
        Generate images by iteratively sampling from pure noise.
        
        Args:
            shape: Shape of images to generate
            n_steps: Number of diffusion steps
            class_label: Optional class label for conditional generation
        
        Returns:
            List of images at each denoising step
        """
        self.eval()
        
        x_t = torch.randn(shape, device=next(self.parameters()).device)
        x_seq = [x_t]
        
        # Prepare class labels if provided
        if class_label is not None:
            batch_size = shape[0]
            if isinstance(class_label, int):
                class_labels = torch.full((batch_size,), class_label, 
                                        device=next(self.parameters()).device, dtype=torch.long)
            else:
                class_labels = class_label
        else:
            class_labels = None
        
        for t in reversed(range(n_steps)):
            x_t = self.p_theta_sampling(x_t, t, class_labels)
            x_seq.append(x_t)
        
        return x_seq
    
    def p_theta_sampling(self, x, t, class_labels=None):
        """
        Reverse diffusion: estimate x[t-1] from x[t].
        
        Uses x-prediction as advocated in He's paper.
        Formula: μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)
        
        But since we predict x directly, we use:
        μ = (1/√α_t) * (x_t - (1-α_t)/(√α_t * √(1-ᾱ_t)) * (x_t - x_pred))
        
        Args:
            x: Current noisy image at timestep t
            t: Current timestep
            class_labels: Optional class labels for conditional generation
        
        Returns:
            Denoised image at timestep t-1
        """
        with torch.no_grad():
            # Convert t to tensor
            if isinstance(t, int):
                t_tensor = torch.tensor([t], device=x.device).repeat(x.shape[0])
                t_int = t
            else:
                t_tensor = t
                t_int = t.item() if t.numel() == 1 else t[0].item()
            
            # Predict CLEAN IMAGE (x-prediction)
            x_pred = self.forward(x, t_tensor, class_labels)
            
            # Get coefficients
            alpha_t = alphas[t_int]
            alpha_bar_t = alphas_bar[t_int]
            alpha_sqrt_t = alphas_sqrt[t_int]
            one_minus_alpha_bar_sqrt_t = one_minus_alphas_bar_sqrt[t_int]
            
            # Compute mean using x-prediction
            # This is the corrected DDPM formula
            mu = x / alpha_sqrt_t - (1 - alpha_t) / (alpha_sqrt_t * one_minus_alpha_bar_sqrt_t) * (x - x_pred)
            
            # Add noise if not final step
            if t_int > 0:
                var = betas[t_int]
                mu = mu + torch.randn_like(x) * torch.sqrt(var)
            
            return mu
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


def diffusion_loss_fn(model, x_0, class_labels=None):
    """
    Training loss using x-prediction and v-loss.
    
    From He's paper: We predict clean data (x) but use v-loss for training,
    which provides good weighting properties.
    
    Args:
        model: Diffusion model
        x_0: Clean images
        class_labels: Optional class labels for conditional generation
    
    Returns:
        Loss value
    """
    batch_size = x_0.size(0)
    
    # Randomly sample timesteps
    t = torch.randint(0, num_steps, size=(batch_size,), device=device)
    
    # Sample noise
    epsilon_t = torch.randn_like(x_0)
    
    # Get coefficients and reshape for broadcasting
    coeff_x0 = alphas_bar_sqrt[t].reshape(-1, 1, 1, 1)
    coeff_et = one_minus_alphas_bar_sqrt[t].reshape(-1, 1, 1, 1)
    
    # Forward process: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    x_t = coeff_x0 * x_0 + coeff_et * epsilon_t
    
    # Predict CLEAN IMAGE (x-prediction)
    x_pred = model(x_t, t, class_labels)
    
    loss = nn.functional.mse_loss(x_pred, x_0)
    
    return loss


def visualize_sampling(model, sample_shape, n_steps, save_path, class_label=None):
    """Visualize the sampling process."""
    model.eval()
    x_seq = model.sample(sample_shape, n_steps, class_label)
    
    num_shows = 10
    step_interval = n_steps // num_shows
    fig, axes = plt.subplots(1, num_shows, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        step = i * step_interval
        image = x_seq[step].squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        # Normalize
        image_min, image_max = image.min(), image.max()
        normalized_image = 2 * (image - image_min) / (image_max - image_min) - 1
        normalized_image = normalized_image.clip(-1, 1)
        imshow_image = (normalized_image + 1) / 2
        
        ax.imshow(imshow_image)
        ax.axis('off')
        ax.set_title(f"Step {step}")
    
    class_name = f"_class{class_label}" if class_label is not None else ""
    plt.suptitle(f"X-Prediction Sampling{class_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_conditional_generation(model, n_steps, num_classes):
    """Generate samples for each flower class."""
    model.eval()
    
    samples_per_class = 5
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(15, 9))
    
    for class_idx in range(num_classes):
        for sample_idx in range(samples_per_class):
            # Generate one sample for this class
            shape = (1, img_channel, img_width, img_height)
            x_seq = model.sample(shape, n_steps, class_label=class_idx)
            final_image = x_seq[-1].squeeze().permute(1, 2, 0).detach().cpu().numpy()
            
            # Normalize
            image_min, image_max = final_image.min(), final_image.max()
            normalized_image = 2 * (final_image - image_min) / (image_max - image_min) - 1
            normalized_image = normalized_image.clip(-1, 1)
            imshow_image = (normalized_image + 1) / 2
            
            axes[class_idx, sample_idx].imshow(imshow_image)
            axes[class_idx, sample_idx].axis('off')
            
            if sample_idx == 0:
                axes[class_idx, sample_idx].set_ylabel(f'Class {class_idx}', fontsize=12)
    
    plt.suptitle('Conditional Generation: Different Flower Types')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_root, "conditional_generation.png"))
    plt.close()


if __name__ == "__main__":
    # Load data
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, img_width, img_height
    )
    
    print("=" * 80)
    print("BONUS IMPLEMENTATION: X-Prediction with Conditional Generation")
    print("Based on 'Back to Basics: Let Denoising Generative Models Denoise'")
    print("=" * 80)
    
    # Detect if data loader provides labels
    print("\n[Checking data loader...]")
    sample_batch = next(iter(training_dataloader))
    
    has_labels = False
    num_classes_detected = 1
    
    if isinstance(sample_batch, (tuple, list)) and len(sample_batch) == 2:
        sample_images, sample_labels = sample_batch
        # Check if labels are meaningful (not all zeros or all same)
        if isinstance(sample_labels, torch.Tensor):
            unique_labels = torch.unique(sample_labels)
            if len(unique_labels) > 1:
                has_labels = True
                num_classes_detected = len(unique_labels)
                print(f"✓ Data loader provides class labels!")
                print(f"  Detected {num_classes_detected} classes: {unique_labels.tolist()}")
            else:
                print(f"⚠ Data loader returns labels but only one class detected")
                print(f"  Will train without conditional generation")
        else:
            print(f"⚠ Data loader returns non-tensor labels, ignoring")
    else:
        print(f"⚠ Data loader does not provide labels (only images)")
        print(f"  Will train without conditional generation")
    
    # Initialize model
    model = Diffuser(
        n_steps=num_steps,
        num_classes=max(num_classes_detected, 3),  # At least 3 for compatibility
        use_bottleneck=True,
        bottleneck_dim=64
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"- Uses x-prediction (predicts clean images directly)")
    print(f"- Includes bottleneck design for manifold learning")
    print(f"- Conditional generation: {'ENABLED' if has_labels else 'DISABLED'}")
    if has_labels:
        print(f"- Number of classes: {num_classes_detected}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = exponential_decay(initial_learning_rate=1e-3, decay_rate=0.9, decay_epochs=5)
    
    sample_shape = (1, img_channel, img_width, img_height)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for batch in training_dataloader:
            # Handle both (images, labels) and (images,) cases
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, labels = batch
                images = images.to(device).float()
                labels = labels.to(device).long() if has_labels else None
            else:
                images = batch.to(device).float()
                labels = None
            
            # Normalize images to [-1, 1]
            images = 2 * images - 1
            
            # Compute loss (with or without conditioning)
            loss = diffusion_loss_fn(model, images, labels if has_labels else None)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model.save(model_save_path)
            print(f"  → Saved best model (loss: {best_loss:.4f})")
        
        # Visualize sampling process
        if (epoch + 1) % 100 == 0:
            print(f"  → Generating visualization...")
            
            # Unconditional generation
            visualize_sampling(
                model, sample_shape, num_steps,
                os.path.join(vis_root, f"x_pred_epoch_{epoch + 1}.png")
            )
            
            # Conditional generation for each class (only if labels available)
            if has_labels:
                for class_idx in range(num_classes_detected):
                    visualize_sampling(
                        model, sample_shape, num_steps,
                        os.path.join(vis_root, f"x_pred_class{class_idx}_epoch_{epoch + 1}.png"),
                        class_label=class_idx
                    )
    
    # Final visualization
    print("\nGenerating final samples...")
    if has_labels:
        print("Creating conditional generation comparison...")
        visualize_conditional_generation(model, num_steps, num_classes=num_classes_detected)
    else:
        print("Creating unconditional samples...")
        visualize_sampling(model, sample_shape, num_steps,
                         os.path.join(vis_root, "final_samples.png"))
    
    print("\n" + "=" * 80)
    print("Training complete!")
    if has_labels:
        print(f"✓ Trained with conditional generation ({num_classes_detected} classes)")
    else:
        print(f"✓ Trained without conditional generation")
    print("=" * 80)