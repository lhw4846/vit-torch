import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        Constructor.

        Parameters
        ----------
        img_size: Size of the input image.
        patch_size: Size of the image patch.
        in_chans: Number of the channels of the input image.
        embed_dim: Dimension of the embedding vector.
                   It should be matched the dimension of the input on the model.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Project the input image into patch embeddings (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # Flatten the embedding (B, embed_dim, N)
        x = x.transpose(1, 2)  # Transpose to match the required shape (B, N, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Constructor.

        Parameters
        ----------
        dim: Dimension of the input embedding.
             It is the dimension of input and output on Transformer.
        num_heads: Number of the head ofr multi-head attention.
                   Each head trains the input embedding respectively.
        qkv_bias: Whether use the bias to generate Query, Key and Value vectors or not.
        attn_drop: Ratio of the dropout for attention score.
        proj_drop: Ratio of the dropout for final output.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # Dimension of each head.
        self.scale = head_dim ** -0.5  # Scale value for normalization of attention score.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Linear function to generate Query, Key and Value vectors.
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout for attention score
        self.proj = nn.Linear(dim, dim)  # Linear function to project original dimension from output of multi-head attention.
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout for final output.

    def forward(self, x):
        B, N, C = x.shape  # (batch size, number of patches, embedding dimension)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Calculate attention scores
        attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
        attn = self.attn_drop(attn)  # Apply dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # Project the output
        x = self.proj_drop(x)  # Apply dropout
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        """
        Constructor.

        Parameters
        ----------
        drop_prob: Probability of drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x / keep_prob * random_tensor
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        """
        Constructor.

        Parameters
        ----------
        dim: Dimension of input embedding.
             It is same as dimension of input and output on self-attention and mlp.
        num_heads: Number of head for multi-head attention.
        mlp_ratio: Ratio for extension of mlp layer.
                   It's hidden layer is generally defined as (dim * mlp_ratio).
        qkv_bias: Whether use the bias to generate Query, Key and Value or not.
        drop: Ratio of dropout for attention and mlp.
        attn_drop: Ratio of dropout for attention score.
        drop_path: Ratio to apply Stochastic Depth.
                   It can be used with the residual connection for solving overfit.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # Layer normalization
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)  # Layout normalization
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # Apply attention with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # Apply MLP with residual connection
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Class token for classification tasks
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)  # Dropout for positional embeddings

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])  # Stack of Transformer blocks
        self.norm = nn.LayerNorm(embed_dim)  # Final layer normalization

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # Classification head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # Patch embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand class token for the batch
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate class token with patch embeddings
        x = x + self.pos_embed  # Add positional embeddings
        x = self.pos_drop(x)  # Apply dropout

        for blk in self.blocks:
            x = blk(x)  # Apply each Transformer black

        x = self.norm(x)  # Final normalization
        x = self.head(x[:, 0])  # Classification head on class token
        return x


if __name__ == "__main__":
    # Create model
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=1)  # Example: setting num_classes=1 for heart rate estimation

    # Example input
    dummy_input = torch.randn(8, 3, 224, 224)  # (batch size, channels, height, width
    output = model(dummy_input)
    print(output.shape, output)
