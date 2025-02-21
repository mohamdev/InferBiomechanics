import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------
# 1. Define VQ-VAE components: Encoder, VectorQuantizer, Decoder
# -----------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # A simple MLP example; for sequences, consider RNN/Transformer layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        h = F.relu(self.fc1(x))
        z = self.fc2(h)  # shape: (batch_size, latent_dim)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z_q):
        # z_q shape: (batch_size, latent_dim)
        h = F.relu(self.fc1(z_q))
        x_recon = self.fc2(h)  # shape: (batch_size, output_dim)
        return x_recon

class VectorQuantizer(nn.Module):
    """
    VQ layer: quantizes the continuous latents by finding the nearest
    embedding in the codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: learnable embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
    
    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        1. Compute distances to all codebook embeddings
        2. Find nearest embedding indices
        3. Get quantized vectors z_q
        4. Compute VQ losses (commitment + codebook)
        """
        # Flatten the latents (in this simple example, latents are already [B, D])
        flat_z = z.view(-1, self.embedding_dim)  # shape: (B, embedding_dim)
        
        # Compute distances from z to each embedding
        #   shape of embedding.weight: (num_embeddings, embedding_dim)
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        
        # Find nearest embedding index for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)  # shape: (B,)
        
        # Quantize and unflatten
        z_q = self.embedding(encoding_indices)  # shape: (B, embedding_dim)
        z_q = z_q.view(z.shape)  # same shape as z, i.e. (batch_size, latent_dim)
        
        # Compute commitment loss: encourages encoder output z to match z_q
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        
        # To update the codebook embeddings, we treat z as "stop gradient"
        #   and pass gradients only through z_q. So we do a trick:
        z_q = z + (z_q - z).detach()
        
        return z_q, commitment_loss, encoding_indices

# -----------------------------------------------------------
# 2. Putting it all together: VQ-VAE model
# -----------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        """
        x shape: (batch_size, input_dim)
        returns:
            x_recon -> reconstructed input
            recon_loss -> e.g. MSE between x and x_recon
            vq_loss -> sum of codebook & commitment losses
        """
        # 1. Encode
        z = self.encoder(x)  # (B, latent_dim)
        
        # 2. Vector Quantization
        z_q, commitment_loss, _ = self.vq(z)
        
        # 3. Decode
        x_recon = self.decoder(z_q)  # (B, input_dim)
        
        # 4. Reconstruction loss (MSE or L1)
        recon_loss = F.mse_loss(x_recon, x)
        
        # We also track codebook loss, but typically that is handled inside the VQ class
        total_loss = recon_loss + commitment_loss
        
        return x_recon, recon_loss, commitment_loss, total_loss

# -----------------------------------------------------------
# 3. Example training loop
# -----------------------------------------------------------
def train_vqvae(model, data_loader, optimizer, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_commit = 0.0
        
        for batch in data_loader:
            # Suppose batch has shape: (batch_size, input_dim)
            x = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, recon_loss, commit_loss, total_loss = model(x)
            
            # Backprop
            total_loss.backward()
            optimizer.step()
            
            epoch_recon += recon_loss.item()
            epoch_commit += commit_loss.item()
        
        # Print average losses
        avg_recon = epoch_recon / len(data_loader)
        avg_commit = epoch_commit / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Recon Loss: {avg_recon:.4f}, Commitment Loss: {avg_commit:.4f}")

# -----------------------------------------------------------
# 4. Dummy data example
# -----------------------------------------------------------
if __name__ == "__main__":
    # Let's create a dummy dataset of random posture vectors:
    # e.g., each posture is 30 joint angles, range ~[-1, 1]
    input_dim = 30
    num_samples = 10000
    random_data = np.random.uniform(-1, 1, size=(num_samples, input_dim))
    
    # Convert to tensor
    dataset = torch.tensor(random_data, dtype=torch.float32)
    
    # Simple DataLoader
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create a VQ-VAE model
    hidden_dim = 128
    latent_dim = 16
    num_embeddings = 64
    commitment_cost = 0.25
    
    model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    train_vqvae(model, data_loader, optimizer, epochs=5, device='cpu')
