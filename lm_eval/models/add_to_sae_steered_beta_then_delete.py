import einops
# Andre was working on Matthew's folders, and Matthew didn't want to edit the same doc at the same time.
def steering_hook_projection(
    activations,#: Float[Tensor],  # Float[Tensor, "batch pos d_in"], Either jaxtyping or lm-evaluation-harness' precommit git script hate a type hint here.
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by finding the projection of each activations, 
    along the specified feature and adding some multiple of that projection to the activation.
    """
    bad_feature =  sae.W_dec[latent_idx] # batch, pos, d_in @ d_in, d_embedding -> batch, pos, d_embedding
    dot_products = einops.einsum(activations, bad_feature, "batch pos d_embedding, d_embedding -> batch pos")
    dot_products /= bad_feature.norm()
    
    # Calculate the projection of activations onto the feature direction
    projection = einops.einsum(
        dot_products, 
        bad_feature, 
        "batch pos, d_embedding -> batch pos d_embedding"
    )
    
    # Add scaled projection to original activations
    return activations + steering_coefficient * projection
    