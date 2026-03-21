from vit import *

class IJepa(nn.Module):
    """
    Image-based Joint-Embedding Predictive Architecture (I-JEPA).

    Given a context image x and a target image y (typically different augmented views
    of the same image), the model encodes both with a shared context encoder, then uses
    a predictor to estimate the target patch representations from the context
    representations. Only target-masked patch positions are returned so the caller
    can compute a loss directly on those positions without extra masking logic.
    """

    def __init__(self, img_size, patch_size=16, n_embed=768, n_head=12, n_layers=12, img_channels=3):
        """
        Args:
            img_size (int): Square input image size (e.g. 224 for ImageNet).
            patch_size (int): Square patch size (default: 16).
            n_embed (int): Embedding dimension (default: 768).
            n_head (int): Number of attention heads (default: 12).
            n_layers (int): Number of Transformer encoder blocks (default: 12).
            img_channels (int): Number of input image channels (default: 3 for RGB).
        """
        super().__init__()

        self.context_encoder = ViTContextEncoder(img_size, patch_size, n_embed, n_head, n_layers, img_channels)

        num_patches = (img_size // patch_size) ** 2 
        self.predictor = ViTPredictor(num_patches, n_embed, n_head, n_layers)
        
        self.n_embed = n_embed


    def forward(self, x, y, target_indices):
        """
        Forward pass computing predicted and actual target patch embeddings.

        Args:
            x (torch.Tensor): Context image of shape (batch_size, channels, height, width).
            y (torch.Tensor): Target image of shape (batch_size, channels, height, width).
            target_indices (torch.Tensor): Binary mask of shape (batch_size, num_patches)
                where 1 marks the target (to-be-predicted) patch positions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - predicted_y_target: Predicted embeddings at target positions,
                  shape (batch_size, num_patches, n_embed), zeros at non-target positions.
                - actual_y_target: Actual encoded embeddings at target positions,
                  shape (batch_size, num_patches, n_embed), zeros at non-target positions.
        """
        # target_indices starts as a binary vector (B, P) marking which patches to predict.
        # It is expanded to (B, P, D) so it can act as a multiplicative mask over embeddings —
        # multiplying by 1 keeps a patch, multiplying by 0 zeros it out. This same expanded
        # mask is passed into the predictor, where it gates the learned mask token injection
        # (see ViTPredictor.forward for how the mask token is applied at target positions).
        target_indices  = target_indices.unsqueeze(-1).expand(-1, -1, self.n_embed)



        encoded_x = self.context_encoder(x)
        encoded_y = self.context_encoder(y)


        predicted_y = self.predictor(encoded_x, target_indices)


        # Zero out non-target positions in both outputs so the caller can compute the loss
        # directly as a sum/mean over the full tensor without needing to index into targets.
        predicted_y_target_mask_embedding = predicted_y*target_indices
        actual_y_target_mask_embedding = encoded_y*target_indices



        return predicted_y_target_mask_embedding, actual_y_target_mask_embedding
