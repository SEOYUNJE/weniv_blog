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



