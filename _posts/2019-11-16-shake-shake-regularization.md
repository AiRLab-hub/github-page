---
layout: post
current: post
cover:  assets/images/posts/2019-11-16-shake-shake-regularization/cover.PNG
navigation: True
title: "Shake-Shake regularization"
date: 2019-11-16 02:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---

### Shake-Shake regularization 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 **"Shake-Shake regularization"** 입니다. 

이 논문은 크게보면 Data augmentation에 속하는 논문 입니다. 지금까지는 Data augmentation을 이미지에 했다면, Shake-Shake regularization 논문은 Internal Representations에 augmentation을 합니다. (Internal Representations은 feature map과, weight를 의미합니다.) 이런 Internal Representations augmentation을 통하여 그 당시 CIFAR10, CIFAR100에서 state of the art를 달성합니다. 

## Motivation
 
![Figure1](/assets/images/posts/2019-11-16-shake-shake-regularization/img1.PNG)
위 그림과 같이 컴퓨터는 사람과는 다르게 물체를 일정 규칙이 있는 스칼라 값으로 인식합니다. 그렇다면 컴퓨터 입장에서는 feature map 그리고 이미지 는 일정 규칙이 있는 데이터이기 때문에 현재처럼 이미지에서만 Data augmentation을 하지말고, Internal Representations에도 Data augmentation을 해주자고 주장합니다. 또한 Internal Representations에 Data augmentation을 해주면 stochastically "blending" 효과가 있다고 주장합니다.

##  Model description on 3-branch ResNets

![Figure2](/assets/images/posts/2019-11-16-shake-shake-regularization/img2.PNG)
![Figure3](/assets/images/posts/2019-11-16-shake-shake-regularization/img3.PNG)
논문 저자는 Shake-Shake regularization을 적용하기 간편한 ResNet 구조에서 실험을 하고, 이해를 돕기 위하여 위와 같은 간단한 수식을 말합니다. 위에 보이는 수식 1은 일반적은 resnet입니다. W는 weight이고, x는 텐서, F는 residual function입니다. 논문 저자는 수식 2와 같이, F앞에 일정 α(0과 1 사이의 랜덤한 값)을 곱해줘 Internal Representations에 Data augmentation을 해주는 효과를 냅니다.

##  Training procedure

![Figure4](/assets/images/posts/2019-11-16-shake-shake-regularization/cover.PNG)

위 그림은 shake-shake-regularization의 전체 학습 절차 입니다. 학습의 forward 부분에서도 α[0~1] 사이의 값을 곱해줘서 data augmentation 효과를 주고, backward 부분에도 β[0~1]의 값을 곱해줘서 기존의 연구 되었던 gradient noise를 주는 방식을 gradient augmentaion으로 대체해 줄 수 있다고 논문 저자는 주장합니다. 또한 테스트시에는 0.5로 값을 고정해 줍니다.

## CIFAR-10, CIFAR-100

![Figure5](/assets/images/posts/2019-11-16-shake-shake-regularization/img4.PNG)

위 그림과 같이 본 논문의 저자는 다양한 실험을 진행하였습니다. Forward pass는 Even / Shake, Backward pass는 Even / Shake / Keep, Mini-batch update rule 은 Image / Batch로 진행하였습니다. Even은 α값을 0.5로 고정하는 방법이고, Shake는 α값은 0~1사이의 값을 랜덤하고 곱해줍니다. Backward의 Kepp은 Forward에 사용한 α값을 그대로 사용하는것 입니다. 또한 Image는 미니배치에서 모든 이미지에 α,β 를 적용하는것이고, Batch 는 미니 배치 단위로 다 같은 α,β를 적용하는것 입니다. 위 그림과 같이 shake shake image가 가장 성능이 좋았고, Even shake batch가 가장 성능이 안 좋았습니다. shake shake image 가 가장 많은 augmentation을 적용한 것이니 당연한 결과라고 생각하고 있습니다. 
 
![Figure6](/assets/images/posts/2019-11-16-shake-shake-regularization/img5.PNG)

위 그림은 배치사이즈를 128에서 32로 줄인 다음 CIFAR100에서 실험한 테이블 입니다. 위에 보이는 결과와 같이 배치사이즈가 줄어 들때는 S S I 조합 보다는 S E I 조합이 조금 더 경쟁력 있다는 것을 알 수 있습니다. 논문저자도 이와 관련해서 해석을 하지 못했습니다. 저 또한 그 이유를 해석하는데 어려움이 있어 혹시 아시는분은 댓글 부탁 드립니다. ㅠㅡㅠ

## Comparisons with state-of-the-art results

![Figure7](/assets/images/posts/2019-11-16-shake-shake-regularization/img6.PNG)

위 표는 기존의 state-of-the-art 결과와 비교한 표 입니다. 보이시는 것과 같이 저자가 주장한 방법이 state-of-the-art를 달성하였습니다. CIFAR10 에서는 S S I 가 CIFAR100에는 S E I 가 가장 좋았습니다. 이 결과를 분석해보면, gradient에 β를 곱해주는 방법은 좋은 성능을 보장하지 못합니다. 제 생각에는 저자가 주장하는 방법은 gradient noise방법을 완벽하게 대체하지는 못하는 것 같습니다. 또한 논문 저나는 imagenet실험을 하지 않아 많이 아쉽습니다.(제 생각에는 imagenet에서는 잘 안될것 같긴 합니다.)

## Correlation between residual branches

![Figure8](/assets/images/posts/2019-11-16-shake-shake-regularization/img7.PNG)

마지막으로 논문저자는 Shake-Shake regularization을 분석합니다. residual branches사이의 공분산을 구하여 서로의 관계성을 계산합니다. 위 그림은 feature map을 펼친후 나온 스칼라 값들로 관계성을 구한것 입니다. 거의 아무것도 안해준 조합인 Even Even Batch 와 거의 모든 augmentation을 해준 Shake Shake Image 조합을 비교 하였습니다. 그 결과, Shake Shake Image 조합에서 residual branches 값들 사이의 관계성이 작게 나왔으므로, Shake-Shake regularization을 해준다면 overfitting이 일어날 확률을 낮춰줄 수 있음을 보여줍니다. 

## 후기

![Figure9](/assets/images/posts/2019-11-16-shake-shake-regularization/img8.PNG)

위 그림과 같이 사실 이 논문 저자는 실험을 상당히 많이 했습니다. BatchNorm을 사용하지 않거나, skip connection을 제거하는등 많은 실험을 하였습니다. 하지만 본 리뷰에서 그런 실험을 리뷰하지 않는 이유는 너무나도 당연하기 때문입니다. BatchNorm을 사용하면 성능저하가 있고, skip connection을 제거하면 성능의 약간의 향상은 있지만 유의미 하지는 않습니다. 또한 그 행동이 CIFAR10,100이여서 상승한 것이지 imagenet이었다면 어떻게 될지 모른다는 생각을 했습니다. 이 논문 자체로는 유의미하게 좋은 논문이라고 생각들진 않지만, shake drop, mixup, cutmix등 다른 augmentation 논문을 읽을때 큰 도움이 된다고 생각이 됩니다. (또한 이 논문이 쓸 내용이 없을떄 어떻게 논문을 꽉 꽉 채울수 있나 좋은 교본으로 느껴졌습니다 ㅋㅋㅋ)



