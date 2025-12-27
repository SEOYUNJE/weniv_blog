# Global Context Vision Transformer (GC ViT)

NVIDIA has recently published (20 June 2022) its papaer, **`GCViT: Global Context Vision Transformer`** which outperforms **ConvNext** and **SwinTransformer**

<ins/>Global Context Vision Transformer 논문</ins> [Click](https://arxiv.org/pdf/2206.09959)

## _Imagenet Top-1 Accuracy(%)_

![image](/develop_blog/img/gcvit_performance.JPG)

## Introduction

GCVit가 가지고 있는 특징 중 Vision Transformer랑 Swin Transformer와의 가장 큰 차이점은 Global Window Attention과 CNN 모듈을 사용한다는 점이다. 먼저 Vision Transformer은 level이 하나이므로 multi-level 정보가 부족하여 segmentation, object detection을 하는데 어려움이 있다. 그에 반해 Swin Transformer의 경우 dense prediction이 가능하지만 local window attention으로 인해 장거리 픽셀 간 의존성 학습이 부족하다. 이에 global query를 이용해 short-range & long-range 정보를 모두 가지도록 Nvida lab실에서 설계한 아키텍처이다. 

![image](/develop_blog/img/gcvit_architecture.JPG)

## ImageNet Benchmarks 

**`ImageNet 1K`** : <ins/>_클래스 수 : 1,000, 이미지 수: 약 1.2M_</ins>


**`ImageNet 21K`**: <ins/>_클래스 수 : 21,000, 이미지 수: 약 14M_</ins>


- ImageNet-1K Pretrained Models

| Model Variant | Acc@1 | #Params(M) | FLOPs(G) |
| ------------- | ----- | ---------- | -------- |
| GC ViT-XXT    | 79.9  | 12 | 2.1 |
| GC ViT-XT    | 82.0  | 20 | 2.6 |
| GC ViT-T    | 83.5  | 28 | 4.7 |
| GC ViT-T2    | 83.7  | 34 | 5.5 |
| GC ViT-S    | 84.3  | 51 | 8.5 |
| GC ViT-S2    | 84.8  | 68 | 10.7 |
| GC ViT-B   | 85.0  | 90 | 14.8 |
| GC ViT-L    | 85.7  | 201 | 32.6 |



- ImageNet-21K Pretrained Models


| Model Variant | Resolution | Acc@1 | #Params(M) | FLOPs(G) |
| ------------- | ----- | ---------- | ---------- | -------- |
| GC ViT-L  | 224X224  | 79.9  | 12 | 2.1 |
| GC ViT-L  | 384X384  | 82.0  | 20 | 2.6 |
| GC ViT-L  | 512X512  | 83.5  | 28 | 4.7 |

## Comparision with Transformer Model

| Model | Query Tokens | Key-Value Tokens | Attention Type | Attention Coverage | 
| ----- | ------------ | ---------------- | -------------- | --------- |
| ViT   | image | image | `self-attention` | ![global](https://img.shields.io/badge/global-blue?style=flat-square) |
| SwinTransformer | window | window | `self-attention` | ![local](https://img.shields.io/badge/local-red?style=flat-square) |
| GCViT | resized-image | window | `image-window attention` | ![global](https://img.shields.io/badge/global-blue?style=flat-square) |

## ViT & Swin Transformer의 한계점 

<img src="https://raw.githubusercontent.com/awsaf49/gcvit-tf/main/image/vit_gif.gif" width=600>

Vistion Transformer(`ViT`)는 전역적인 문맥 정보를 효과적으로 학습할 수 있다는 장점이 있지만, self-attnetion의 계산 복잡도가 O(n2)로 증가한다는 구조적 한계를 가진다 이로 인해 고해상도 입력을 다루는데 많은 연산 비용이 요구되며, Multi Scale 정보를 명시적으로 활용하지 못하다는 점에서 한계를 보인다. 
이러한 특성은 픽셀 단위의 **dense prediction**이 필요한 **object detection**나 **semantic segmenation**과 같은 컴퓨터 비전 태스크에서 ViT를 범용적인 아키텍처로 사용하기 어렵게 만든다.

![img](/develop_blog/img/swin.JPG)


그러나 Swin Transformer 역시 한계를 가진다. local window 기반 attention은 receptvie field가 제한적이기 때문에 장거리 의존성을 충분히 포착하기 어렵다. window shifting과 같은 cross-window 연결 방식 또한 각 window 인접 영역의 제한된 범위만을 cover할뿐, 전역적인 문맥 정보를 효과적으로 통합하는 데에는 한계가 존재한다.

## GCViT의 특징 

1. _GCViT는 Swin Transformer처럼 `window attention`을 기법을 적용한다_

2. _기존 Swin Transformer의 경우 `local self-attention`과 `shifted-window self-attention`을 통해 short-range dependices만 학습한 반면, `global-window-attnetion`을 적용하여 short-range, long-range dependices를 모두 학습한다_

3. _`global-window-attention`도 window-attention의 일종이지만 query로 window가 아닌 imgae를 사용해 `long range information`을 포착한다_

4. _ViT랑 Swin Transformer과 달리, `cnn module`도 적용하여 `inductive bias`를 사용한다_

## GCViT의 Argument

| args | description |  gcvit_xxtiny | gcvit_xtiny | gcvit_tiny | gcvit_small | gcvit_base | 
| ---- | ----------- |  ------------ | ----------- | ---------- | ----------- | ---------- |
| `embed_dim` | feature size dimension | 64 | 64| 64 | 96 | 128 |
| `depths` | number of layers in each stage | (2, 2, 6, 2) | (3, 4, 6, 5)| (3, 4, 19, 5) | (3, 4, 19, 5) | (3, 4, 19, 5) |
| `window_size` | window size in each stage | (7,7,14,7) | (7,7,14,7) | (7,7,14,7) | (7,7,14,7) | (7,7,14,7) |
| `mlp_ratio` | MLP ratio | 3 | 3 | 3 | 2 | 2 |
| `num_heads` | number of heads in each stage | (2, 4, 8, 16) | (2, 4, 8, 16) | (2, 4, 8, 16)| (3, 6, 12, 24) | (4, 8, 16, 32) |
| `qkv_bias` | bool argument for query key, value learnable bias| True | True | True | True | True |
| `drop_rate` | dropout rate | 0 | 0 | 0 | 0 | 0 |
| `attn_drop` | attention dropout rate| 0 | 0 | 0 | 0 | 0 |

## GCViT 적용에 따른 Feature Map Shape 변환

- **Img Size**: 224X224
- **GCViT Model**: gcvit_tiny
- **window size**: (7,7,14,7)
- **embed_dim**: 64,
- **detphs**: (3, 4, 19, 5)
- **num_heads**: (2, 4, 8, 16)

| Layer | Output Shape | Descriptions |
| ----- | ------------ | ------------ |
| Input Layer | (None 224, 224, 3) |  (Batch, Height, Width, Channel) | 
| PatchEmbed | (None, 56, 56, 64) | (Batch, Height/Window, Width/Window, embed_dim |
| Pos Drop | (None, 56, 56, 64) | None |
| ![Level 0](https://img.shields.io/badge/Level_0-1E90FF?style=flat-square) | (None, 28, 28, 128) | (Batch, Height/Window * 2, Width/Window *2, embed_dim * 2) |
| ![Level 1](https://img.shields.io/badge/Level_1-4DA3FF?style=flat-square) | (None, 14, 14, 256) | (Batch, Height/Window * 4, Width/Window *4, embed_dim * 4) |
| ![Level 2](https://img.shields.io/badge/Level_2-9370DB?style=flat-square) | (None, 7, 7, 512) | (Batch, Height/Window * 8, Width/Window *8, embed_dim * 8)|
| ![Level 3](https://img.shields.io/badge/Level_3-6A0DAD?style=flat-square) | (None, 7, 7, 512) | 마지막 level에선 downsample을 적용하지 않는다|
| Norm | (None, 7, 7, 512) | Layer Normalization | 
| Pool | (None, 512) | GlobalAveragePooling2D | 
| Head | (None, 1) | num_classes = 1 |

## GCViT 내 사용하는 CNN Module

### 1. SE(Squeeze-Excitation) Block

SE는 채널 Attention 기법 중 하나로 GCViT에선 MBconv block을 사용하며 해당 conv block에서 SEBlock을 적용한다. 

```python
class SE(nn.Module):
    """
        Squeeze and Excitation Block
    """
    def __init__(
        self, 
        in_chs: int, 
        rd_ratio: float = 0.25)
        
        """
        Args:
            in_chs: input features dimension.
            rd_ratio: reduce ratio.
        """

        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_chs, int(in_chs * rd_ratio), bias=False),
            nn.GELU(),
            nn.Linear(int(in_chs * rd_ratio), in_chs, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b,c)
        out = self.fc(out).view(b,c,1,1)

        return x * out

```

### 2. ReduceSize

해당 ReduceSize는 GCViT Level 마지막에 위치한 모듈로 keep_dim=False인 경우에만 채널을 2배 늘리고, 마지막 level의 경우에는 적용하지 않는다

![img](/develop_blog/img/reducesize.JPG)

```
    input shpae: (B, H, W, C) <- feature map
    output shape: 
        if keep_dim: (B, H//2, W//2, C) <- general use
        else: (B, H//2, W//2, 2C) <- last gcvit level
```

- **MB Conv Block**: `Depthwise Conv` -> `GELU` -> `SE` -> `Proj Conv`

- **Call**: `Layer Norm` -> `MBConv` -> `Reduction` -> `Layer Norm`

```python
    class ReduceSize(nn.Module):
        def __init__(
            self, 
            dim: int,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            keep_dim: bool = False,
        ):
        super().__init__()

        self.mb_conv = nn.Sequential(
            nn.Conv(dim, dim, kernel_size=3, stride=1, padding=1, group=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2,
                                   padding=1, bias=False)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)
        
    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0,3,1,2) # (B,H,W,C) -> (B,C,H,W)
        x = x + self.mb_conv(x) # Skip Connection
        x = self.reduction(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.norm2(x)
        return (x)
```


## GCVit - PatchEmbed

해당 GCVit 아키텍처에선 Patchembedding을 Linear projection이 아닌 Conv Stem 방식으로 진행한다. Linear Projection의 경우 patch들이 서로 독립된 정보를 가지고 있는 반면 conv stem의 픽셀들의 경우 서로 인접한 경우 정보가 연관되어 있어 학습을 하는데 있어 훨씬 빠르고 안정적으로 가능하다

- ViT의 Patch Embedding: `Linear projection` 적용
    - p: patch_size, D: embed_dim, N: num_patches
    - rearrange -> nn.Linear(P²C → D)
    - (B, H, W, C) -> (B, N, D) 

- GCViT의 Patch Embedding: `Conv Stem` 적용 
    - input shape: (B, in_chs, H, W)
    - output shape: (B, H//4, W//4, dim)
    - `nn.Conv2d(in_chs, dim, k=3,s=2)` -> `ReduceSize(keep_dim=True)`


```python
class PatchEmbed(nn.Module):
    def __init__(self, 
                in_chs: int = 3, 
                dim: int = 96):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1)
        self.down_conv = ReduceSize(dim=dim, keep_dim=True)
    
    def forward(self, x):
        x = self.proj_conv(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.down_conv(x)
        return x

```

## GCVit - Window Partition, Window Reverse

### Window Partition

Window Attention을 적용하기 위해선 주어진 Feature Map을 window size로 변환을 해줘야 한다

```
    intput shape: (B, H, W, C) <- Feature map
    output shape: (B * num_windows, window_size, window_size, C)
```

```python
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)

    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)

    return windows
```

### Window Reverse

반대로 Local or Global Window Attention을 적용한 이후에는 ... ReduceSize 적용을 위해서 다시 (B,H,W,C) 형태로 돌려놔야 한다 

```
    input shape: (B*num_windows, window_size**2, C)
    output shape: (B, H, W, C)
```

```python
def window_reverse(windows, window_size, H, W):

    C = windows.shape[-1]
    x = windows.view(-1, H//window_size, W//window_size, window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1,H,W,C)
    return x

```

## GCVit - how to generate global query

Local Window Attention의 경우 query, key, value 모두 window를 사용하지만

장거리 의존성 학습을 위해 GCViT에선 Swin Transformer의 `shifted local attention` 대신해서 `global window attention` 개념을 사용했다

이에 Global Window Attention에선 query가 global query gen에서 생성한 feature map을 이용한다

이때, global q는 각 GCViT Level에서 받은 Feature map을 가지고 window 크기로 resize하는 과정을 거치고 하나의 global query를 모든 local window에 query로 제공하며 되므로 효율적이다. 

### 1. FeatExtract

해당 모듈은 `MBconv`과 `Maxpool`로 이루어진 간단한 CNN 모듈이다 

그리고 해당 FeatExtract을 반복해서 window_size 크기만큼 resize를 진행한다

```
    input shape: (B,D,H,W)
    output shape: 
        if keep_dim: (B,D,H,W)
        else: (B,D,H//2,W//2)
```

```python
    class FeatExtract(nn.Module):
    
    """
        Resize: Feature map -> window for global attention
        k번 Repeat(MBConv -> MaxPool 2X2)
        k번: log2(feature map Height // window_size)
    """
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.keep_dim = keep_dim
    
    def forward(self, x):
        x = x.contiguous()
        x = x = self.conv(x) # skip connection
        if not self.keep_dim:
            x = self.pool(x)
            
        return x
```

### 2. GlobalQueryGen

FeatExtract를 반복하여 해당 level의 window_size랑 동일하게 해당 level의 feature map을 resize를 진행한다. 그렇게 해서 나온 resized image가 global window attention에 적용하는 global query or q_global이다.

이때 FeatExtract의 반복 횟수: log2(Feature map Height // window_size[0])

```
    input shape: (B, D, H, W)
    output shape: (B,1,num_heads,window_size*window_size,dim_head)
```

```python
class GlobalQueryGen(nn.Module):
    
    def __init__(self,
                 dim: int,
                 input_resolution: int | float,
                 image_resolution: int | float,
                 window_size: int,
                 num_heads: int):
        super().__init__()
        
        ## Level 0: (B, H//4, W//4, C)
        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )
            
        ## Level 1: (B, H//8, W//8, 2C)
        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )
        ## Level 2: (B, H//16, W//16, 4C)
        elif input_resolution == image_resolution//16:
            if window_size == input_resolution:
                self.to_g_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )
            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )
        
        ## Level 3: (B, H//32, W//32, 8C)
        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )
        
        self.resolution = input_resolution
        self.num_heads = num_heads 
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')            
    
    def forward(self, x):
        x = self.to_q_qlobal(x) # (B,C,H,W) -> (B,C,window,window)
        x = x.permute(0,2,3,1) # (B,C,window,window) -> (B,window,window,C)
        x = x.contiguous().view(x.shape[0],1,self.N,self.num_heads,self.dim_head).permute(0,1,3,2,4)
        return x 
```

## GCVit - Window Attention

### Local Window Attention

Local Window Attention은 Vision Transformer의 Multi-Head Self-Attention과 구조적으로 거의 동일하다.

다만 positional encoding 방식에서 차이가 있다. ViT는 입력 패치 토큰에 절대적 positional embedding을 더하는 방식을 사용하며, 이는 embedding dimension이 고정되어 있을 때 효과적으로 학습된다. 반면 GCViT는 stage(level)가 증가함에 따라 embedding dimension이 점진적으로 확장되므로, 동일한 방식의 절대 positional embedding을 적용하기 어렵다.
이에 GCViT에서는 relative_position_bias를 사용하여, query–key 간 attention score를 계산한 뒤 softmax를 적용하기 이전 단계에서 bias를 추가함으로써 상대적인 위치 정보를 학습한다.


```
    input shape: (B*num_windows, window_size**2, dim)
    output shape: (B*num_windows, window_size*2, dim)
```

- `q,k attn` -> `attn + relative_position_bias` -> `qk, v attn` -> `proj` -> `proj_drop`

```python
    class LocalWindowAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        
        # Window 토큰 사이의 상대적 h 거리: -(h-1) ~ 0 ~ (h-1)
        # Window 토큰 사이의 상대적 w 거리: -(w-1) ~ 0 ~ (w-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0]) # h의 token 위치 [0,1,2]
        coords_w = torch.arange(self.window_size[1]) # w의 token 위치 [0,1,2]
        # h 좌표: [[0,0,0],[1,1,1],[2,2,2]]
        # w 좌표: [[0,1,2],[0,1,2],[0,1,2]]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, w,w)
        # h: [0,0,0,1,1,1,2,2,2] 
        # w: [0,1,2,0,1,2,0,1,2]
        coords_flatten = torch.flatten(coords, 1) # (2,w*w)
        # window 토큰 사이 거리 구하기: (2, w*w, w*w)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (w*w, w*w, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1 # 음수 제거, 범위[0~2w-1]
        relative_coords[:, :, 1] += self.window_size[1] - 1 # 음수 제거, 범위[0~2w-1]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # h_idx = h_dix * width
        relative_position_index = relative_coords.sum(-1) # idx = h_idx * width + w_idx, (w*w, w*w)
        
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B, N, C = x.shape # (B*num_windows, window_size**2, dim)
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        
        # ((2W-1)*(2W-1), num_heads) -> (w*w, w*w, num_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # (w*w, w*w, num_hads) -> (num_heads, N, N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### Global Window Attention 

global window attention과 local window attention의 차이점은 단순하다

query 값으로 window가 아니라 Global Query Gen에서 나온 q_global을 입력값으로 준다. 다만, q_global를 num_windows만큼 repeat해 각 local window에 적용하면 되고 하나의 global query를 모든 local window에 적용하다는 점에서 계산이 효율적이다. 

또한 local의 경우 qkv를 위해 `nn.Linear(dim, dim*3, bias=qkv_biase)`를 했지만 global은 외부에서 query를 가져오므로 `nn.Linear(dim, dim*2, bias=qkv_bias)`로 k,v만 구한다

- `q_global X (num_windows)` -> `q_global, k attn` -> `attn + relative_position_bias` -> `qk & attn` -> `proj` -> `proj_drop` 

```
    inp: (B*num_windows, window_size**2, dim)
        q_query: (B,1,num_heads,window_size**2, head_dim) 
               -> (B, num_windows, num_heads, window_size**2, head_dim)
        out: (B*num_windows,window_size**2, dim)
```

```python
class GlobalWindowAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape # (B_*num_windows, window_size**2, dim)
        B = q_global.shape[0] # B
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor') # num_windows
        # (B_, N, 2, num_heads, head_dim) -> (2, B_, num_heads, N, head_dim)
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2,0,3,1,4)        
        k, v = kv[0], kv[1]
        # (B, num_windows, num_heads, window_size**2, head_dim)
        q_global = q_global.repeat(1, B_dim, 1, 1, 1) # Compute Coefficient
        # (B, num_windows, num_heads, window_size**2, head_dim)
        # (B * num_windows, num_heads, window_size**2, head_dim)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1)) # (B_, num_heads, N, N)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

## _GCVit Block_ (Key Argument)

**note**: GCViT Block에서 localwindowattention, globalwindowattention이 모두 다 적용되는 것이 아니라 Block에는 local or global 중에서 하나만 적용된다. 그러다 보니 기본적으로 block의 갯수는 최소 2개 이상이고 

이때 block의 갯수는 gcvit의 hyperparameter에서 depth로 설정가능하다

| model_name | level 0 | level 1 | level 2 | level 3 |
| ---------- | ------- | ------- | ------- | ------- |
| gcvit_xxtiny |   2   |    2    |    6    |    2    |

또한 ViT의 Transformer Block과 GCViT Block의 또 다른 차이점은 Drop Path의 사용 여부이다. GCViT는 여러 level에서 local 및 global attention을 반복적으로 적용하기 때문에 residual network의 깊이가 증가하며, 이로 인해 overfitting이 발생할 가능성이 높다.
이를 완화하기 위해 GCViT는 **Drop Path**와 **layer_scale** 기법을 도입하여 학습 안정성과 일반화 성능을 조정한다.

```
        input shape: (B,H,W,C)
        output shape: (B,H,W,C)
        
        if attention: WindowAttention
            Local Window Attention
        else attention: WindowAttentionGlobal
            Global Window Attention
```

| Layer | Input Shape | OutPut Shape | 
| ----- | ----------- | ------------ | 
| ReduceSize(keep_dim=False) | (B,H,W,C) | (B,H//2,W//2,2C) |
| Layer Norm | (B, H//2, W//2, 2C) | (B, H//2, W//2, 2C) |
| Window Partition | (B*num_w, w, w, 2C| (B*num_w, w**2, 2C) |
| window attention | (B*num_w, w**2, 2C) | (B*num_w, w**2, 2c) | 
| window reverse | (B*num_w, w**2, 2C) | (B, H//2, W//2, 2C ) |
| residual | (B, H//2, W//2, 2C ) | (B, H//2, W//2, 2C ) | 
| Layer Norm | (B, H//2, W//2, 2C ) | (B, H//2, W//2, 2C ) |
| Mlp | (B, H//2, W//2, 2C ) | (B, H//2, W//2, 2C)|
| residual | (B, H//2, W//2, 2C) | (B, H//2, W//2, 2C) |

```python
class GCViTBlock(nn.Module):
    def __init__(self,
                dim: int,
                input_resolution: int | float,
                num_heads: int,
                window_size: int = 7,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                qk_scale: Optional[float] = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                act_layer: Type[nn.Module] = nn.GELU,
                attention=GlobalWindowAttention,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                layer_scale: int | float | None = None):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.attn = attention(
            dim,
            num_heads = num_heads,
            window_size = window_size,
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop = attn_drop,
            proj_drop = drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_dims=dim, hidden_dims=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.layer_scale = False
        if layer_scale is not None: 
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        
    def forward(self, x, q_global):
        B , H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # (B*num_windows, window_size, window_size, dim)
        x_windows = window_partition(x, self.window_size) 
        # (B*num_windows, window_size, window_size, dim) -> (B*num_windows, window_size**2, dim)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global) 
        # (B*num_windows, window_size, window_size, dim) -> (B, H, W, dim)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        
        return x
```

## _GCVit Level_ (Key Argument)

GCVit Level에서 가장 중요한 건 GCVit Block이며 해당 module은 **GCVit Block**, **Global Query Gen**, **ReduceSize**이다. 

이전 Level에서 출력한 Feature map에서 GCVit Block을 반복해 local attention, global query gen에서 얻은 q_global로 학습하는 global attention을 적용해 short-range, long-range 학습을 진행하고 맨 마지막에 Reduce Size를 적용하는 형식이다. 

- `Feature map` -> `GCViT Block X N번` -> `Reduce Size`

```
    input shape: (B, H, W, D)
    output shape:
        if keep_dim: (B, H//2, W//2, D)
        else: (B, H//2, W//2, 2D)
```

## _GCViT_

위의 모든 Sub module을 하나씩 bottom up 방식으로 진행하면서 분석해보았다.

그럼 지금까지 살펴본 sub module의 input, output shape를 정리해보자

|  sub module | Input | Output |
|  ---------- | ----- | ------ |
|   ReduceSize | (B,H,W,C) | (B,H//2,W//2,2C) |   
|   PatchEmbed | (B,in_chs,H,W) | (B,H//4,W//4,D) | 
|   FeatExtract |  (B,C,H,W) | (B,C,H,W) | 
|  GlobalQueryGen | (B,C,H,W) | (B,1,num_heads,window_size**2,head_dim)
|  Window Partition | (B,H,W,C) | (B*num_windows, window_size, C) |
|  Window Reverse | (B*num_windows, window_size**2, C) | ( B, H, W, C) | 
|  LocalWindowAttention |  (B*num_windows, window_size**2, D) | (B*num_windows, window_size**2, D) |
|  GlobalWindowAttention | (B*num_windows, window_size**2, D) | (B*num_windows, window_size**2, D) |
|  GCVit Block | (B,H,W,C) | (B,H,W,C) |
|  GCVit Level | (B,H,W,C) | (B,H//2,W//2,2C)
|  GCVit | (B, in_chs, H, W) | (B, num_classes) |

그리고 중요한 점은 GCViT tiny, small, Base, Large 모두 Level 0~4까지 사용했다는 점.. 그리고 맨 마지막 level에선 Reduce Size를 적용하지 않았다는 점을 유의 하자.

또한 Level3, Level 4부터는 window size랑 resized image랑 크기를 인위적으로 동일하게 만들어서 max pool을 적용하지 않았다.
이는 window_size가 [7,7,7,7]이 아니라 [7,7,14,7]인 이유이다. 

이미지 크기가 224이고 embed_dim이 32라고 가정해보자 

| Level | 입력 Feature map | window_size | Description | 
| ----- | ---------------- | ----------- | ---------- |
| `Level 0` | (B, 56, 56, 32) |    7   |  ![Feature > Window](https://img.shields.io/badge/Feature_>_Window-red?style=flat-square)  |
| `Level 1` | (B, 28, 28, 64) |   7   | ![Feature > Window](https://img.shields.io/badge/Feature_>_Window-red?style=flat-square) |
| `Level 2` | (B, 14, 14, 128) |   14   |  ![Feature = Window](https://img.shields.io/badge/Feature_=_Window-blue?style=flat-square) | 
| `Level 3` | (B, 7, 7, 256) |  7   | ![Feature = Window](https://img.shields.io/badge/Feature_=_Window-blue?style=flat-square)  |



