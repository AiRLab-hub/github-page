---
layout: post
current: post
cover:  assets/images/posts/2019-11-13-Transfer-learning/cover.png 
navigation: True
title: Transfer learning
date: 2019-11-12 20:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: hyeoncheol
---


Transfer learning 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 노현철 입니다. 
제가 이번에 리뷰할 논문은 **"How transferable are features in deep neuralnetworks?"** 이른바 **"transfer learning"** 에 관한 논문입니다.


### Introduction

transfer learning이란 
적은 dataset에서 학습을 시키면 over-fitting이 일어날 가능성이 많아짐으로 많은 dataset에서 학습을 시킨 일부의 layer들을 가져와서 이 적은 dataset에 적용 시켜 적은 dataset에서도 강인함을 보여줄 수 있는 것이 transfer learning입니다. 

![Figure1](/assets/images/posts/2019-11-13-Transfer-learning/01.png)

모든 데이터셋으로 학습을 시키면 초반 레이어에서는 Generality한 파라미터들이 나오고 후반 레이어에서는 Specificity한 파라미터들이 나온다. transfer learning은 초반 레이어에 나오는 Generality한 파라미터를 이용하는 것이다. 그 이유는 각각의 데이터 셋을 학습시켜 얻고자하는  목적이나 원하는 것들이 다르기 때문에 Generality한 파라미터들을 사용하는 것이다. 이 논문에서는 어디까지가 Generality하고 어디서부터 Specificity한지 실험도 하였다.

![Figure2](/assets/images/posts/2019-11-13-Transfer-learning/02.png)

그림에서 보듯이  Generality 레이어에서는 Gabor filters, color blobs 와 같은 것들이 학습된다.
![Figure3](/assets/images/posts/2019-11-13-Transfer-learning/03.png)
##### Gabor filters

![Figure4](/assets/images/posts/2019-11-13-Transfer-learning/04.png)
##### color blobs


### Experimental

● 1000개의 이미지넷에서 500(A), 500(B) 로 나누어 실험을 한다.

● A, B 모두 총 8개의 conv레이어를 사용할 것이다.

● transferred layers는 frozen시키거나 fine-tuned시키는 실험이다. fine-tuned을 사용한 것은 (+)가 쓰여져 있는 것이다.

![Figure5](/assets/images/posts/2019-11-13-Transfer-learning/05.png)
그림에서 보듯이 baseA 와 baseB는 transfer없이 학습을 시킨 것이고 BnB, AnB는 각각 B에서 학습시킨 파라미터를 B에 적용, A에서 학습시킨 파라미터를 B에 적용한다는 의미이다. (+)가 붙은 경우들은 fine-tuned을 시킨경우들이다. (기본적으로 transferred layers들은 frozen 시킴) 
transfer시키는 레이어의 범위는 1번째 레이어 에서부터 마지막 8번째 레이어 까지 각각 실험하여서 어디까지 General 한지보고, 또한 fine-tuning 하였을 때 성능의 변화가 있는지 보는 실험이다.
#### results
![Figure6](/assets/images/posts/2019-11-13-Transfer-learning/06.png)
#### BnB 
1. (n=1, 2) base B와 동일

2. (n=3,4,5) 성능저하
- 레이어간 co-adapted(동화기능)이 있어 상위 레이어만 이 기능을 배울 수 없다.

3. (n=6,7) 성능 다소상향
- 학습의 필요성이 점차 줄어듦.
- (6,7or7,8)사이 features가 co-adapted이 덜하다.
- 중간보다 하단,상단이 optimization 하기 좋다.

#### BnB+ 
1. 전체적으로 base B와 비슷
- co-adapted(동화기능)의 성능저하를 방지시켜줌.

2. 성능향상은 없었음.

#### AnB
1. 일부 layer는 Transfer Learning 하는것이 좋다.
2. (n=1,2) layer는 general 하다.
3. (n=3) 약간감소, (n=4,5,6,7) 성능대폭하락
  - 이것으로 인해 두 가지 감소 이유를 알 수 있음. 

    1) (n=3,4,5) co-adaptation으로 인한 감소
 
    2) (n=6,7) generality 보단 specificity 하기 때문

#### AnB+
1. Fine-tuning은 좋은 성능을 보여줌.
- Transfer Learning의 목적과 반대로 dataset이 많은 경우에도 사용하면 성능을 향상 시킬 수 있다!
2. (n=1~7) 성능을 유지, 성능 향상
- 이 효과는 첫번째 네트워크(A)의 양에 크게 의존하지 않음.
- 이 효과는 너무 많은 retraining을 한다는 것은 놀라운 일.


#### Conclusions

* co-adapted 때문에 중간layers에서 분할하는 것은 어렵다.
  
* 상위layers의 specialization로 인해 Transfer Learning의 부정적인 영향을 끼친다.
  
* random weights보단 Transfer Learning이 좋다.
  
* new task에 Transfer Learning, fine-tuning을 더하면 성능을 향상 시키는데 유용할 수 있다.


### 후기

랩실에 들어오고 첫 세미나 발표를 한 논문이고, 또한 끝까지 읽어본 몇 안되는 논문이라 더욱 뜻깊고 기억에 남는 논문이였습니다.
지금은 많이 부족하지만 세미나와 논문리뷰를 하면서 실력을 한층 한층 쌓아 나아가겠습니다. 감사합니다!!!!!
