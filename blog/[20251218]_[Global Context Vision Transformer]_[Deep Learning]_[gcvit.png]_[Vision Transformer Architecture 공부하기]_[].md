# Global Context Vision Transformer (GC ViT)

NVIDIA has recently published (20 June 2022) its papaer, **`GCViT: Global Context Vision Transformer`** which outperforms **ConvNext** and **SwinTransformer**

<ins/>Global Context Vision Transformer 논문</ins> [Click](https://arxiv.org/pdf/2206.09959)

## _Imagenet Top-1 Accuracy(%)_

![image](/develop_blog/img/gcvit_performance.JPG)

## Introduction

GC ViT leverages global context self-attention modules, joint with local self-attention, to effectively yet efficiently model both long and short range spatial interactions, without the need for expensive operations such as computing attention masks or shifting local windows.

![image](/develop_blog/img/gcvit_architecture.JPG)

## ImageNet Benchmarks 

![label](https://img.shields.io/badge/ImageNet_1K-2496ED) : <ins/>_클래스 수 : 1,000, 이미지 수: 약 1.2M_</ins>



![label](https://img.shields.io/badge/ImageNet_21K-2496ED): <ins/>_클래스 수 : 21,000, 이미지 수: 약 14M_</ins>


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

