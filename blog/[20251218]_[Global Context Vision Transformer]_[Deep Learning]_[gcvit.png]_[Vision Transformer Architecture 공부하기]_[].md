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

![image](/develop_blog/img/swin.JPG)


그러나 Swin Transformer 역시 한계를 가진다. local window 기반 attention은 receptvie field가 제한적이기 때문에 장거리 의존성을 충분히 포착하기 어렵다. window shifting과 같은 cross-window 연결 방식 또한 각 window 인접 영역의 제한된 범위만을 cover할뿐, 전역적인 문맥 정보를 효과적으로 통합하는 데에는 한계가 존재한다.

