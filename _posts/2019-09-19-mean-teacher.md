---
layout: post
current: post
cover:  assets/images/posts/2019-09-19-mean-teacher/cover.png
navigation: True
title: Mean teachers are better role models
date: 2019-09-19 02:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: jaemin
---

Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 이재민입니다!

오늘 소개할 논문은 Mean teachers are better role models ([arXiv:1703.01780](https://arxiv.org/abs/1703.01780))이며, NIPS 2017에서 소개된 논문입니다.

이 논문은 Semi-Supervised Leaning에 관련된 논문이고, Π Model과 Temporal ensemble 이후에 소개되었습니다.

<img src="/assets/images/posts/2019-09-19-mean-teacher/figure1.png" width="80%" alt="figure1" />

#### Π Model
Π Model은 위 그림의 위 쪽과 같은 구조와 같이, 라벨이 주어진 경우 Closs Entropy를 사용하여 학습을 하고, 라벨이 주어지지 않은 경우는 augmented input의 결과와 이전과 다른 dropout 과 augmention을 사용한 결과의 squared difference를 이용해 학습을 합니다.

#### Temporal ensemble
Temporal ensemble과 Π Model의 차이점은 라벨이 주어지지 않았을 때 Temporal ensemble은 이전의 결과에 대한 값들을 앙상블한 결과를 가지고 학습을 하는 것입니다.

### Mean Teacher
<img src="/assets/images/posts/2019-09-19-mean-teacher/cover.png" width="80%" alt="figure3" />
위에서 소개한 두개의 모델은 모델을 공유하여 학습이 되지만, Mean Teacher의 경우 Student와 Teacher 모델로 나뉘어 학습이 진행됩니다. Teacher 모델은 Student 모델의 Weight를 공유하고, 대신에 Student모델의 아래의 방식과 같이 EMS Weight를 사용합니다.

<img src="/assets/images/posts/2019-09-19-mean-teacher/figure2.png" width="30%" alt="figure2" />

그 후 라벨이 주어지지 않은 경우 기존 방식과 유사하게 Teacher 모델과 Student 모델의 consistency cost (J)를 계산하여 학습이 진행되게 됩니다.
<img src="/assets/images/posts/2019-09-19-mean-teacher/figure3.png" width="40%" alt="figure3" />

이러한 방법은 Temporal ensemble에 비하여 2가지 이점을 가지는데, 첫 번째는 Student와 Teacher 사이의 빠른 Feedback이 가능하여 더 높은 ACC를 가지게 되는 것이고, 두 번째는 large scale datasets을 online으로 학습할 수 있는 것 입니다.

<img src="/assets/images/posts/2019-09-19-mean-teacher/table1.png" width="80%" alt="table1" />

위의 결과와 같이 SVHM과 CIFAR10 에서의 Mean Teacher를 사용한 네트워크가 Π Model과 Temporal ensemble보다 더 높은 퍼포먼스를 보임을 확인할 수 있습니다.

## References
- Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." Advances in neural information processing systems. 2017.

- Rasmus, Antti, et al. "Semi-supervised learning with ladder networks." Advances in neural information processing systems. 2015.

- Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).
