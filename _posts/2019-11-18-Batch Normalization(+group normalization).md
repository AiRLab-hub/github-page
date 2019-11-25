---
layout: post
current: post
cover:  assets/images/posts/2019-11-18-Batch Normalization(+group normalization)/cover.png
navigation: True
title: Batch Normalization-Accelerating Deep Network Training by Reducing Internal Covariate Shift
date: 2019-11-18 10:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: hyeyun
---

Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 강혜윤입니다.

제가 이번에 읽은 논문은 **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**(Batch Norm) ([arXiv:1605.07146](https://arxiv.org/pdf/1502.03167.pdf))이며 2015년 발표가 된 연구입니다. 


### [Abstract]

- DNN 학습 시키는 것이 어려운 이유 : 학습을 시키는 과정에서 이전 layer의 parameter 변화로 다음 layer의 input의 분포가 변화함 (weight 값의 변화)  <br>
  => “internal covariate shift”

- Batch normalization에서 normalization을 모델 아키텍처의 일부로 만들고, 각 mini-batch에 대해 normalization을 수행

- Batch normalization의 장점 : <br>
  ① 높은 learning rate 사용 가능 <br>

  ② 초기화에 크게 신경 쓰지 않아도 됨<br>

  ③ regularizer의 역할로 몇몇 경우에 Dropout을 사용하지 않아도 됨 

### [Introduction]
▶ SGD(Stochastic Gradient Descent)

 ① Gradient Descent : loss function을 계산할 때 전체 data에 대해 계산, 계산 량이 가장 많음<br>
 (배치는 전체 데이터 셋라고 가정)<br>

 ② Mini-batch Gradient Descent : training data의 배치(batches)만 이용해서 그라디언트(gradient)를 구하는 것<br>

 ③ Stochastic Gradient Descent : loss function을 계산할 때 일부 data에 대해 계산 (mini-batch), 계산 량을 줄임<br>
 (확률적(Stochastic) : 용어는 각 배치를 포함하는 하나의 예가 무작위로 선택된다는 것을 의미)<br>
 미니배치(mini-batch)가 데이터 한 개로 이루어짐<br>

- DNN을 효율적으로 학습시킬 수 있도록 함<br>
- SGD의 변형인 momentum, Adagrad 등도 있음<br>
- loss값을 최소화하기 위해 parameterθ을 최적화시킴<br>

![Batch Normalization](/assets/images/posts/2019-11-18-BN/1.png)

- SGD 사용 시 특징<br>

 ① Stochastic Gradient는 실제 Gradient의 추정 값이며 이것은 미니배치의 크기 N이 커질수록 더 정확한 추정 값을 가지게 된다.<br>
 ② 미니배치를 뽑아서 연산을 수행하기 때문에 최신 컴퓨팅 플랫폼에 의하여 병렬적인 연산 수행이 가능하여 더욱 효율적이다.<br>

- 기존의 문제점<br>

 ① neural network를 학습하기 위한 hyper parameter들의 초기 값 설정을 굉장히 신중하게 해줘야 함 (그렇지 않으면 covariate shift가 발생할 수 있음)

 ② Saturation problem<br>
  기울기가 0이 되어 weight 업데이트량이 0이 되는 현상<br>
  sigmoid 함수에서 큰 값을 가지면 기울기가 0이 되어 사라짐<br>
  -> ReLU 함수 사용과 신중하게 initialization하고 작은 learning rate를 사용하면서 해결 (완전한 해결책은 아님)

- Batch Normalization의 등장<br>

 ① internal covariate shift 문제를 줄일 수 있음<br>

 ② DNN 학습을 가속화 시킬 수 있음<br>

[batch normalization 공식]

![Batch Normalization](/assets/images/posts/2019-11-18-BN/2.png)

 - mini batch 단위에서 정규화 수행<br>
 - Mini batch 내의 한 example 내에서의 Activation 들은 각각 독립적<br>
 - Γ: scale 조정, β: shift 조정<br>

### [Experiments_MNIST]

![Batch Normalization](/assets/images/posts/2019-11-18-BN/3.png)

 ⦁Batch Normalization makes the distribution more stable and reduces the internal covariate shift.


[Experiments_ImageNet classification]

![Batch Normalization](/assets/images/posts/2019-11-18-BN/6.png)

### [Group normalization과 Batch normalization 비교]

Normalization formulation
![Batch Normalization](/assets/images/posts/2019-11-18-BN/9.png)

① Batch normalization
![Batch Normalization](/assets/images/posts/2019-11-18-BN/10.png)
 -> batch 단위로 정규화 수행 <br>
 -> batch norm의 문제점 : batch size 단위로 정규화 시키는 과정에서 batch size에 의존적이게 되고, batch size가 작을 경우 학습이 잘 이루어지지 않음 <br>

② Group normalization
![Batch Normalization](/assets/images/posts/2019-11-18-BN/11.png)

 위와 같은 문제점을 해결하기 위해 정규화를 batch 단위로 하지 않고, channel을 그룹으로 나누어 그 그룹을 단위로 정규화 수행<br>
 (더 이상 batch size에 의존적이지 않음)

![Batch Normalization](/assets/images/posts/2019-11-18-BN/12.png)

 -> 위에 실험을 보면 batch size에 의존적인 BN은 batch size 별로 학습에 차이를 보였고, batch size에 비의존적인 GN은 어떤 batch size에도 학습이 잘 이루어지는 것을 확인할 수 있다.