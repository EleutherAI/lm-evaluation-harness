import torch

def batch_vector_projection(vectors: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects each vector in a batch onto a target vector.
    
    Args:
        vectors: Tensor of shape (b, p, d) where:
                b is the batch size
                p is the number of vectors per batch
                d is the dimension of each vector
        target: Tensor of shape (d,) - the vector to project onto
                
    Returns:
        Tensor of shape (b, p, d) containing the projected vectors
        
    Example:
        b, p, d = 32, 10, 3  # batch of 32, 10 vectors each, in 3D
        vectors = torch.randn(b, p, d)
        target = torch.randn(d)
        projections = batch_vector_projection(vectors, target)
    """
    # Ensure target is unit vector
    target = torch.nn.functional.normalize(target, dim=0)
    
    # Reshape target to (1, 1, d) for broadcasting
    target_reshaped = target.view(1, 1, -1)
    
    # Compute dot product between each vector and target
    # Result shape: (b, p, 1)
    dot_products = torch.sum(vectors * target_reshaped, dim=-1, keepdim=True)
    
    # Project each vector onto target
    # Multiply dot products by target vector
    # Result shape: (b, p, d)
    projections = dot_products * target_reshaped
    
    return projections, dot_products

# Test function
if __name__ == "__main__":
    # Create sample data
    batch_size, vectors_per_batch, dim = 2, 3, 4
    vectors = torch.randn(batch_size, vectors_per_batch, dim)
    target = torch.randn(dim)
    
    # Compute projections
    projected, dot_products = batch_vector_projection(vectors, target)
    
    _, zero_dot_products = batch_vector_projection(vectors - projected, target)
    assert torch.allclose(zero_dot_products, torch.zeros_like(zero_dot_products), atol=1e-6)
    print("Without proj, close to zero")
    # Verify shapes
    print(f"Input shape: {vectors.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Output shape: {projected.shape}")
    