
import torch
from torch import nn
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_heads=12,
                 mlp_size=3072,
                 dropout=0.1):
        super().__init__()

        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

    def forward(self, x):
        return self.block(x)

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 embedding_dim=768):
        super().__init__()
        self.patch_size=patch_size

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Image size must be divisible by patch size | Image shape: {image_resolution} | Patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flatten = self.flatten(x_patched)
        return x_flatten.permute(0,2,1)

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 patch_size:int=16,
                 in_channels:int=3,
                 num_layers:int=12,
                 hidden_size_D:int=768,
                 MLP_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 weight_decay:float=0.03,
                num_classes:int=1000
                 ):
        super().__init__()
        assert img_size % patch_size == 0, f"Image size is not divisible by patch size. Image size: {img_size} | Patch size: {patch_size}"
        self.num_patches = int((img_size**2)/(patch_size**2))
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, hidden_size_D), requires_grad=True)
        self.pos_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, hidden_size_D), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=hidden_size_D)
        self.Transformer_encoder = nn.Sequential(*[TransformerEncoderBlock() for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=hidden_size_D),
            nn.Linear(in_features=hidden_size_D,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.pos_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.Transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
