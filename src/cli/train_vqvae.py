from models.VqVaeBaseline import *


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