---
layout: post
current: post
cover:  assets/images/posts/2019-08-21-Group-Normalization/cover.png 
navigation: True
title: Group Normalization
date: 2019-08-20 20:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---


Group Normalization 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 **"Group Normalization"** 입니다. 

이 논문은 Yuxin Wu와 Kaiming He 씨가, batch의 크기가 어쩔수 없이 작아야 하는 상황(detection, segmentation and video)에서 batch norm의 한계점을 느끼고 이를 개선하는 방법을 제안 합니다.

### Introduction

Batch Normalization은 딥러닝에서 모델을 설계할 때 필수요소처럼 여겨지고 있습니다. 하지만 detection, segmentation, video와 같이 메모리 소비 때문에 어쩔 수 없이 batch의 크기가 제약될 때 Batch Normalization의 오류는 빠르게 증가합니다. 이 논문에서는 Batch Normalization을 대체할 Group Normalization을 제안하고, 다양한 task에서 이를 실험하고 결과를 보여줍니다. 당연히 결과는  Group Normalization 이 좋습니다.

![Figure1](/assets/images/posts/2019-08-21-Group-Normalization/Figure1.png)

### Related Work

Batch Normalization의 치명적인 단점을 해결하기 위한 노력은 과거에도 많이 있었습니다. Batch Normalization은 말 그대로 Batch 단위로 Normalization 하는것 인데 batch 크기에 영향을 많이 받습니다. 그렇기 때문에 논문에서 소개하는 Related Work은 대부분이 채널 단위로 Normalization을 합니다. 아래의 두 식은 일반적으로 사용하는 평균과 표준편차를 구하느 식 입니다. μ 는 평균이고 σ는 표준편차 입니다. i는 feature 입니다. 예를들어  2D 이미지에서 i는 (iN,iC,iH,iW) 이고 4개의 백터(N, C, H, W)를 가지고 있습니다. ε은 아주 작은값 입니다. X  와 평균 X'이 같으면 0이 되기 때문에 이를 방지하기 위한 값 입니다. S 는 평균과 표준편차가 계산된 픽셀의 집합입니다.

앞으로 소개드릴 LN,IN,GN은 BN과 마찬가지로 , 학습 가능한 파라메터 γ,β 가 존재하고, 감마는 스케일 베타는 쉬프트 입니다.

![formul1](/assets/images/posts/2019-08-21-Group-Normalization/formul.png)
![formul2](/assets/images/posts/2019-08-21-Group-Normalization/formul1.png)

#### Layer Norm

![Figure2](/assets/images/posts/2019-08-21-Group-Normalization/Figure2.png)

Layer Normalization은 Feature 차원 Normalization 입니다. batch 단위로 Normalization 하는 것이 아니라 데이터의 한 이터레이션 마다 평균과 표준편차를 구해주는 것 입니다. 아래의 그림과 식을보면 더 이해가 잘 되실 것 같습니다.

![formul3](/assets/images/posts/2019-08-21-Group-Normalization/formul2.png)
![Figure3](/assets/images/posts/2019-08-21-Group-Normalization/Figure3.png)


#### Instance Norm

![Figure4](/assets/images/posts/2019-08-21-Group-Normalization/Figure4.png)

Instance Normalization은 Layer Norm에서 각 채널마다 Normalization 해주는 방법 입니다. style transfer을 위해 고안된 방법이기 때문에 style transfer에서 BN을 대체하여 많이 사용하고, real-time generation에 효과적 이라고 알려져 있습니다.

![formul4](/assets/images/posts/2019-08-21-Group-Normalization/formul4.png)

### Group Normalization

![Figure5](/assets/images/posts/2019-08-21-Group-Normalization/Figure5.png)

Group Normalization은 Layer Norm과 Instance Norm의 중간쯤 이라고 생각하시면 이해하기 편하실꺼 같습니다. 채널을 그룹지어서 그룹단위로 Normalization 하는 방법입니다. 만약 그룹이 채널 전체면 Layer Norm이 되는것이고, 그룹이 채널 하나면 Instance Norm 이 되는것 입니다.

![formul5](/assets/images/posts/2019-08-21-Group-Normalization/formul5.png)

코드로 구현하기 매우 간단합니다.  아래는 pytorch로 구현한 코드 입니다.

```python
import torch
import torch.nn as nn

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num = 32, eps = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        self.group_num = group_num # 전체 채널을 나눌 그룹 숫자입니다.
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1)) # 학습가능한 파라메터 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1)) # 학습가능한 파레메터 beta
        self.eps = eps # 0방지

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)  # 그룹으로 묶고

        mean = x.mean(dim = 2, keepdim = True) # 평균
        std = x.std(dim = 2, keepdim = True) # 표준편차

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W) # 원래대로 돌리기.

        return x * self.gamma + self.beta
```
Group의 개수 또한 유동적으로 변경되는 값입니다. 이 논문에서는 그룹을 32개 만드는게 가장 좋았다고 합니다. 아래 그림에 보이시는것 처럼 그룹이 1 개이면 LN과 같고 그룹당 채널이 1 이면 IN과 같습니다.

![Figure6](/assets/images/posts/2019-08-21-Group-Normalization/Figure6.png)

#### Experiments

아래 그림은 batch 크기가 32 images/GPU에서 BN, LN, IN, GN 의 train error 와 val error의 비교 입니다. 잘 알려져 있는것 처럼 BN이 가장 좋은 성능을 보입니다. 하지만 GN도 BN과 비교해서 그래서 크게 차이가 나는게 아닙니다. 또한 왼쪽 train error 그림을 보시면 GN이 조금더 train error가 작습니다. 이걸로 보아 GN이 최적화 하는 능력이 더 좋다고 볼 수 있습니다. 

![Figure7](/assets/images/posts/2019-08-21-Group-Normalization/Figure7.png)

아래 그림은 batch 크기가 변함의 따라 BN과 GN의 성능 차이를 나타낸 그래프 입니다. BN은 batch 크기가 작아지면 작아질 수록 정확도가 매우 떨어졌으나, GN은 batch 크기가 변함의 따라 정확도의 차이가 거의 없습니다.

![Figure8](/assets/images/posts/2019-08-21-Group-Normalization/Figure8.png)


### 후기

GN은 옛날부터 관심있던 논문인데 이제서야 제대로 읽었습니다.! 이 논문역시 다양한 Normalization 방법을 잘 정리해주고, 의식에 흐름대로 궁금증을 다 실험하고 풀어줘서 너무 좋았습니다.!! Experiments는 제가 중요하다고 생각하는 부분만 설명하고 , 나머지는 생략했기 때문에 더 자세한 내용을 알고 싶으신 분들은 본 논문을 참고하시기 바랍니다.