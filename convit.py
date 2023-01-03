import torch 
import torch.nn as nn 
from torch.nn import functional as F



class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, embed_size, in_channels = 3):
        super().__init__()
        self.img_size = img_size
        self.Patch_size = patch_size
        self.embed_size = embed_size
        self.n_patches = (img_size //patch_size) ** 2 
        self.in_channels = in_channels 

        self.transform = nn.Conv2d(in_channels = in_channels, out_channels = embed_size, kernel_size = patch_size,
                                stride = patch_size)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)

            if isinstance(m.weight, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0 )

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.Weight, 1.0)


    def forward(self, x):

        x = self.transform(x) #(n_samples, embed_dim, height, width)
        x = x.flatten(2)
        x = x.transpose(2, 1)  #(n_samples, n_patches, embed_dim)

        return x 

    

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, drop_mlp):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_mlp)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)

            if isinstance(m.weight, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0 )

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.Weight, 1.0)


    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x 

    
class GPSA(nn.Module):
    def __init__(self, dim, n_heads = 8, proj_drop = 0., attn_drop = 0.,
                qkv_bias = False, locality_strength = 1.0, use_local_init = True):
        super().__init__()
        self.dim = dim 
        self.n_heads = n_heads 
        self.head_dim = dim // n_heads 
        self.scale = n_heads ** -0.5

        self.query = nn.Linear(dim, dim, bias = qkv_bias)
        self.key = nn.Linear(dim, dim, bias = qkv_bias)
        self.value = nn.Linear(dim, dim, bias = qkv_bias)
        self.pos_proj = nn.Linear(3, n_heads)
        self.fc_out = nn.Linear(dim, dim)
        self.fc_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.gating_params = nn.Parameter(torch.ones(self.n_heads))
        self.locality_strength = locality_strength

        self.apply(self._init_weights)

        if use_local_init:
            self.local_init(locality_strentgth = locality_strength)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)

            if isinstance(m.weight, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0 )

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.Weight, 1.0)


    
    def forward(self, x):
        n_samples, n_patches, dim = x.shape
        if not hasattr(self, 'rel_distances') or self.rel_distances.size(1) != n_patches:
            self.get_rel_distance(n_patches)
        
        attention = self.get_attention(x)

        v = self.value(x).reshape(n_samples, n_patches, self.n_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)
    
        x = attention @ v   #(n_smples, n_heads, n_patches, head_dim)
        x = x.transpose(1, 2) #(n_smples, n_patches, n_heads, head_dim)
        x = x.flatten(2)  #(n_samples, n_patches, n_heads*head_dim)
        x = self.fc_out(x)
        x = self.fc_drop(x)

        return x 



    def get_attention(self, x):
        n_samples, n_patches, dim = x.shape
        q = self.query(x).reshape(n_samples, n_patches, self.n_heads, self.head_dim)
        k = self.key(x).reshape(n_samples, n_patches, self.n_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) #(n_samples, n_patches, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        pos_score = self.rel_distances.expand(n_samples, -1, -1, -1) #(n_samples, num_patches, num_patches, 3)
        pos_score = self.pos_proj(pos_score) #(n_samples, num_patches, num_patches, n_heads)
        pos_score = pos_score.permute(0, 3, 1, 2)  #(n_samples, n_heads, num_patches, num_patches)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim = -1)
        pos_score = pos_score.softmax(dim = -1)

        gating = self.gating_params.view(1, -1, 1, 1) #(1, num_heads, 1, 1) one for each head
        attention = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attention /= attention.sum(dim = -1).unsqueeze(-1)
        attention = self.attn_drop(attention)

        return attention



    def local_init(self, locality_strength):
        self.value.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1

        kernel_size = int(self.n_heads ** 0.5)
        center = (kernel_size-1)/2 if kernel_size % 2 == 0 else kernel_size // 2

        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size*h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position, 0] = 2* (h2-center)*locality_distance

        self.pos_proj.weight.data *= locality_strength

    
    def get_rel_distance(self, num_patches):
        img_size = int(num_patches**0.5)
        rel_distances = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim = 0).repeat_interleave(img_size, dim = 1)

        indd = indx**2 + indy**2

        rel_distances[:, :, :, 2] = indd.unsqueeze(0)  #unsqueeze not necessary
        rel_distances[:, :, :, 1] = indy.unsqueeze(0)
        rel_distances[:, :, :, 0] = indx.unsqueeze(0)

        device = self.key.weight.device

        self.rel_distances = rel_distances.to(device)



class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads = 8, proj_drop = 0.0):

        super(SelfAttention, self).__init__()
        self.dim = dim 
        self.n_heads = n_heads 
        self.head_dim = dim // n_heads 
        self.scale = n_heads ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.fc_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)

            if isinstance(m.weight, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0 )

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.Weight, 1.0)


    def forward(self, x):
        n_samples, n_patches, dim = x.shape

        assert dim == self.dim, 'dim should be equal to the dimension declared in the constructor'

        q = self.query(x)  #Each with dim: (n_samples, n_patches + 1, dim)
        k = self.key(x)
        v = self.value(x)


        q = q.reshape(n_samples, n_patches, self.n_heads, self.head_dim) #(n_samples, n_patches, self.n_heads, self.head_dim)
        k = k.reshape(n_samples, n_patches, self.n_heads, self.head_dim)      
        v = v.reshape(n_samples, n_patches, self.n_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) #(n_samples, n_heads, n_patches, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        k_t = k.transpose(-1, -2) #(n_samples, n_heads, head_dim, n_patches)

        weights = (torch.matmul(q, k_t)) * self.scale #(n_samples, n_heads, n_patches, n_patches)

        scores = weights.softmax(dim = -1)

        weighted_avg = scores @ v  #(n_smples, n_heads, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) #(n_smples, n_patches, n_heads, head_dim)

        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches, n_heads*head_dim)

        x = self.fc_out(weighted_avg)
        x = self.fc_drop(x) 

        return x 



class Block(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias = False, mlp_ratio = 4., proj_drop = 0., attn_drop = 0., 
                use_gpsa = True, locality_strength = 1.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.use_gpsa = use_gpsa

        if self.use_gpsa:
            self.attention = GPSA(dim, n_heads = n_heads, qkv_bias = qkv_bias, proj_drop = proj_drop, 
                                attn_drop = attn_drop, locality_strength = locality_strength)
        else:
            self.attention = SelfAttention(dim, n_heads = n_heads, proj_drop = proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features = dim, hidden_features = dim * mlp_ratio, out_features = dim, drop_mlp = proj_drop)


    def forward(self, x):
        x = x + self.attention(self.mlp(self.norm1(x)))
        x = x + self.attention(self.mlp(self.norm2(x)))

        return x



class Convit(nn.Module):
    def __init__(self, img_size = 32, patch_size = 16, n_classes = 10, embed_dim = 768, n_heads = 8, mlp_ratio = 4., 
                qkv_bias = False, drop = 0., attn_drop = 0., local_layers = 10, locality_strength = 1., depth = 12,
                use_pos_embed = True):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.local_layers = local_layers
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size, embed_size = embed_dim)
        self.n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim = embed_dim, n_heads = n_heads, qkv_bias = qkv_bias, mlp_ratio = mlp_ratio, proj_drop = 0.,
            attn_drop = attn_drop, use_gpsa = True, locality_strength = locality_strength) 
            if i < local_layers else
            Block(dim = embed_dim, n_heads = n_heads, qkv_bias = qkv_bias, mlp_ratio = mlp_ratio, proj_drop = 0., 
            attn_drop = attn_drop, use_gpsa = False) 
            for i in range(depth) ])

        self.norm = nn.LayerNorm(embed_dim)     
        self.head = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls, std = 0.02)
        self.head.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = 0.02)

            if isinstance(m.weight, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0 )

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.Weight, 1.0)


    def forward(self, x):
        n_samples, n_patches, embed_dim = x.shape

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)

        if self.use_pos_embed:
            x += self.pos_embed
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            if i == self.local_layers:
                torch.cat((cls_token, x), dim = 1)

            x = block(x)

        cls_embed  = self.norm(x)
        x_cls = self.head(cls_embed)

        return x_cls








