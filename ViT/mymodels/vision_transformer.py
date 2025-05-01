'''
Code borrowed from https://github.com/facebookresearch/convit which uses code from timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

Modifications include adaptation to image reconstruction, variable input sizes, and patch sizes for both dimensions. 
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm import create_model
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath

class VisionTransformer(nn.Module):
    """
    Vision Transformer for fastMRI reconstruction.

    Two modes:
      * pretrained=True: wraps timm pretrained ViT-Base/16
      * pretrained=False: custom ViT-Like scratch model
    """
    def __init__(
        self,
        avrg_img_size=int(320),
        patch_size= int(16),
        in_chans=int(1),
        embed_dim=48,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_pos_embed=True,
        pretrained=True,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.pretrained = pretrained
        self.in_chans = in_chans
        if not isinstance(patch_size, (tuple, list)):
            patch_size = int(patch_size)        
        self.grid_size = (
                    avrg_img_size // self.patch_size[0],
                    avrg_img_size // self.patch_size[1],
        )

        if pretrained:
            # --- timm pretrained ViT ---
            from timm import create_model
            timm_vit = create_model(
                'vit_base_patch16_224', pretrained=True,
                in_chans=in_chans, num_classes=0
            )
            # reuse its patch embedding conv
            self.patch_embed = timm_vit.patch_embed
            # interpolate pos_embed
            pe = timm_vit.pos_embed  # [1, 1+G0*G0, D]
            cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
            D = pe.shape[-1]
            G0 = int(grid_pe.shape[1]**0.5)
            grid_pe = grid_pe.view(1, G0, G0, D).permute(0,3,1,2)
            grid_pe = F.interpolate(grid_pe, size=self.grid_size, mode='bilinear', align_corners=False)
            grid_pe = grid_pe.permute(0,2,3,1).view(1, -1, D)
            self.pos_embed = torch.cat([cls_pe, grid_pe], dim=1)
            self.pos_drop = timm_vit.pos_drop
            self.blocks   = timm_vit.blocks
            self.norm     = timm_vit.norm
            # custom head
            Ph, Pw = self.patch_embed.proj.kernel_size
            P = in_chans * Ph * Pw
            self.head = nn.Linear(timm_vit.embed_dim, P)
            # adjust config
            if hasattr(self.patch_embed, 'img_size'):
                self.patch_embed.img_size = (avrg_img_size, avrg_img_size)
        else:
            # --- custom ViT scratch (e.g. ViT-Large style) ---
            self.depth      = depth
            d_model = embed_dim * num_heads
            self.d_model    = d_model
            # patch embed conv
            self.patch_embed = nn.Conv2d(
                in_chans, d_model,
                kernel_size=patch_size,
                stride=patch_size
            )
            # positional embedding
            self.use_pos_embed = use_pos_embed
            if use_pos_embed:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, d_model, *self.grid_size)
                )
                trunc_normal_(self.pos_embed, std=.02)
            self.pos_drop = nn.Dropout(p=drop_rate)
            # transformer blocks
            dpr = torch.linspace(0, drop_path_rate, depth).tolist()
            self.blocks = nn.ModuleList([
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    use_gpsa=False,
                ) for i in range(depth)
            ])
            self.norm = nn.LayerNorm(d_model)
            # head maps back to image patches
            P = self.patch_size[0] * self.patch_size[1] * in_chans
            self.head = nn.Linear(d_model, P)

    def forward_features(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.patch_embed(x)           # [B, d_model, Hp, Wp]
        Hp, Wp = x.shape[-2:]
        if self.use_pos_embed:
            pe = F.interpolate(self.pos_embed, size=(Hp, Wp), mode='bilinear', align_corners=False)
            x = x + pe
        # flatten
        x = x.flatten(2).transpose(1,2)    # [B, N, d_model]
        x = self.pos_drop(x)
        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x, (Hp, Wp))
        x = self.norm(x)
        return x

    def forward(self, x):
        # ensure 4-D
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feats = self.forward_features(x)
        patches = self.head(feats)        # [B, N, P]
        B, N, _ = patches.shape
        Ph, Pw = self.patch_size
        Hp, Wp = x.shape[-2] // Ph, x.shape[-1] // Pw 

        x = patches.view(B, Hp, Wp, self.in_chans, Ph, Pw)  # [B, Hp, Wp, C, Ph, Pw]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()        # [B, C, Hp, Ph, Wp, Pw]
        x = x.view(B, self.in_chans, Hp * Ph, Wp * Pw)      # [B, C, H, W]
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # choose MHSA
        self.attn = MHSA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            grid_size=None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, grid_size):
        self.attn.current_grid_size = grid_size
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MHSA(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0., proj_drop=0., grid_size=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.current_grid_size = grid_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True, grid_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)       
        self.k = nn.Linear(dim, dim, bias=qkv_bias)    
        self.v = nn.Linear(dim, dim, bias=qkv_bias)       
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(1*torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
        self.current_grid_size = grid_size
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def get_attention(self, x):
        B, N, C = x.shape  

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)            

        pos_score = self.pos_proj(self.rel_indices).expand(B, -1, -1,-1).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1) 
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    def local_init(self, locality_strength=1.):
        
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, ):
        H, W = self.current_grid_size
        N = H*W
        rel_indices = torch.zeros(1, N, N, 3)
        indx = torch.arange(W).view(1,-1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(H, H)
        indy = torch.arange(H).view(1,-1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat_interleave(W, dim=0).repeat_interleave(W, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.v.weight.device
        self.rel_indices = rel_indices.to(device)
        
    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices()

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
