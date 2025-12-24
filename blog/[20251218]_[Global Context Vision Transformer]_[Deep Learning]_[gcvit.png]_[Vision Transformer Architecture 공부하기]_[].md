# Global Context Vision Transformer (GC ViT)

NVIDIA has recently published (20 June 2022) its papaer, **`GCViT: Global Context Vision Transformer`** which outperforms **ConvNext** and **SwinTransformer**

<ins/>Global Context Vision Transformer 논문</ins> [Click](https://arxiv.org/pdf/2206.09959)

## _Imagenet Top-1 Accuracy(%)_

![image](/develop_blog/img/gcvit_performance.JPG)

## Introduction

GC ViT leverages global context self-attention modules, joint with local self-attention, to effectively yet efficiently model both long and short range spatial interactions, without the need for expensive operations such as computing attention masks or shifting local windows.

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

1. SE(Squeeze-Excitation) Block

채널 간 중요도를 학습하는 채널 어텐션 기법의 한 일종

![img](/develop_blog/img/SE.JPG)

```python
class SEBlock(nn.Module):

    def __init__(self, in_chs, rd_ratio = 0.25):
        super(SEBlock, self).__init__()
        self.in_chs = in_chs
        self.ratio = ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(self.in_chs, self.in_chs * self.rd_ratio, 1, bias=False),
            nn.GELU(inplace=True),
            nn.Conv2d(self.in_chs * self.rd_ratio, self.in_chs, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        return x * self.sigmoid(out)
```

2. Fused-MBConv

GCVit에서 사용하는 MBConv 구조는 일반적인 Efficient 계열 모델에서 사용하는 MBConv 구조랑 차이점이 존재한다

먼저, MBConv(Mobile Inverted Bottleneck)의 구조는 

```bash
    r = expand ratio, C`=C이면 residual 연결

    1x1 Conv (Expand, C -> rC)
    -> 3x3 Depthwise Conv (rC -> rC, groups=rC)
    -> Squeeze Excitation
    -> 1x1 Conv (Project, rC -> C')
    -> Resiudal (C == C', stride=1)
```

```python
class MBConvBlock(nn.Module):
    def __init__(self,
                 in_chs: int,
                 out_chs: Optional[int] = None,
                 expand_ratio: float = 1.0,
                 act_layer: Type[nn.Module] = nn.GELU):
        super().__init__()

        out_chs = out_chs or in_chs
        mid_chs = int(expand_ratio * in_chs)

        self.conv_dw = nn.Conv2d(in_chs, mid_chs, kernel_size=3, stride=1, padding=1, groups=in_chs, bias=False)
        self.act = act_layer()
        self.se = SEBlock(in_chs=mid_chs)
        self.conv_pw = nn.Conv2d(mid_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        shortcut = x
        out = self.conv_dw(x)
        out = self.act(out)
        out = self.se(out)
        out = self.conv_pw(out)
        out = out + shortcut

        return out
```

3. DownSample


```python
class Downsample2d(nn.Module):

    """
        LayerNorm: 학습 안정성 Up
        MBconv: 정보 손실 최소화
        Reduction: 학습 가능한 DownSampling
    """

    def __init__(self, 
                in_chs: int, 
                out_chs: Optional[int] = None, 
                act_layer: Type[nn.Module] = nn.GELU,
                norm_layer: Type[nn.Module] = nn.LayerNorm2d):

        super().__init__()
        out_chs = out_chs or in_chs

        self.norm1 = norm_layer(out_chs) if norm_layer is not None else nn.Identity()
        self.conv_block = MbConvBlock(out_chs, act_layer=act_layer)
        self.reduction = nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False)

        self.norm2 = norm_layer(out_chs) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        out = self.norm1(x)
        out = self.conv_block(out)
        out = self.reduction(out)
        out = self.norm2(out)

        return out
```
## Key Component in GCViT

1. Stem(PatchEmbed)

**Note**: 일반적으로 `PatchEmbed`로 명칭하지만 GCViT 논문에선 이를 `Stem`이라고 명칭함

- ViT의 Patch Embedding: `Linear projection` 적용
    - p: patch_size, D: embed_dim
    - rearrange -> nn.Linear(P²C → D)
    - (B, H, W, C) -> (B, H*W/p*p, D) 

- GCViT의 Patch Embedding: `Conv Stem` 적용 
    - Conv2d(k=3,s=2,p=1) -> Downsample2d
    - (B, H, W, C) -> (B, H/4, W/4, D)


아래의 코드는 timm의 gcvit 코드를 참고하여 작성하였다 

```python
class Stem(nn.Module):

    """
       1. linear에 비해 훨씬 안정적으로 학습한다
       2. Inductive Bias를 가지고 있다.(인접한것끼리 연관되어 있음)
       3. 즉 vit랑 swin transformer과는 달리 Overlapping Patches를 가진다
       4. kernel과 stride를 동일하게 하면 non overlapping patch가 된다.

    """

    def __init__(self, in_chs=3, out_chs=embed_dim, act_layer=nn.GELU, norm_layer=LayerNorm2d):
        super().__init__()

        # (B, H, W) -> (B, H/2, W/2)
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1) # padding = (kernel_size - 1) // 2
        
        # (B, H/2, W/2, C) -> (B, H/4, W/4, 2C)
        self.down = Downsample2d(out_chs, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.down(x)
    

```

2. Level

해당 모듈에는 CNN, Transformer 모듈이 모두 포함되어 있다. 

크게 Global Token Gen, Block, Downsample 3가지 part로 나뉘며
작동 순서는 아래와 같다

- `Feature Map` -> `Global Token` -> `Local/Global Window Attention` -> `DownSample`

| Level Number | (B, H, W, 3) |
| ------------ | ------------ |
|   **Level 1**    | (B, H/4, W/4, C) |
|   **Level 2**    | (B, H/8, W/8, 2C) |
|   **Level 3**    | (B, H/16, W/16, 4C) |
|   **Level 4**    | (B, H/32, W/32, 8C) |
