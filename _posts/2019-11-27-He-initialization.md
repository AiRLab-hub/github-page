---
layout: post
current: post
cover:  assets/images/posts/2019-11-27-He-initialization/ReLU_vs_PReLU.PNG
navigation: True
title: Delving Deep into Rectifiers Surpassing Human-Level Performance on Imagenet Classification
date: 2019-11-27 1:30:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: juhui
---
안녕하세요 AiRLab 박주희입니다.
오늘 소개할 논문은 Delving Deep into Rectifiers Surpassing Human-Level Performance on Imagenet Classification (https://arxiv.org/pdf/1502.01852.pdf)이며, ICCV2015에서 소개된 논문입니다.

### Introduction
이 논문에서는 두가지 측면에 대한 image classification을 위한 rectifier neural networks를 연구했습니다.
먼저 ReLU에서 파생된 Parametric Rectified LinearUnit (PReLU)을 제안합니다. PReLU는 추가 계산 cost가 거의 들지 않고, overfitting의 위험도 적습니다.
두번째로 rectifier의 비 선형성을 고려한 강력한 초기화 방법을 도출했습니다. 이 방법은 깊은 모델에서 직접적으로 사용 할 수있고, 더 깊고 넓은 network architecture를 살펴볼수있습니다.

이런 학습가능한 활성함수와 초기화 방법을 통해 ImageNet 2012 classification dataset에서 4.94% top-5 error를 달성하였습니다.
이 결과는 보고된 인간 수준 성능(5.1%)를 능가하는 최초의 결과 입니다.

### Parametric Rectifiers
<img src="/assets/images/posts/2019-11-27-He-initialization/ReLU_vs_PReLU.PNG" width="50%" alt="error"/>
ReLU와 PReLU그래프 입니다. 여기서 PReLU 그래프의 경우, 음의 부분은 일정한 값을 가지지 않고 적응적으로 학습 합니다.
<img src="/assets/images/posts/2019-11-27-He-initialization/PReLU_Definition.PNG" width="30%" alt="error"/>
a<sub>i</sub>=0 일경우 ReLU가 되고, a<sub>i</sub>가 학습 가능한 파라미터일 경우 PReLU가 되며 a<sub>i</sub>=0.01 일 경우 Leaky ReLU가 됩니다.
이때 PReLU는 매우 적은 수의 추가 매개변수를 도입하였습니다. (추가 파라미터의 수는 총 채널수와 동일하고, 이는 총 가중치 수를 고려할때 무시가 가능합니다.)
그렇기 때문에 Overfitting에 대해 걱정을 하지 않아도 되는 이점이 있습니다.

### Initialization of Filter Weights for Rectifiers
<b>“Xavier” initialization VS “He” initialization</b>
<img src="/assets/images/posts/2019-11-27-He-initialization/Xavier_vs_He.PNG" width="50%" alt="error"/>
“Xavier”초기화 방법은 무작위 초기화가 아닌 입력과 출력의 특성을 고려한 방법으로, 선형인 경우에서만 사용 가능하지만
"He"초기화 방법은 비선형일 경우에도 사용이 가능한 "Xavier"방법의 변형입니다. 이 방법은 비 선형적인 ReLU와 PReLU함수에서도 사용이 가능합니다.
위 그래프의 빨간색이 "He"초기화 방법을 파란색은 "Xavier"초기화 방법을 나타냅니다.
He 초기화 방법은 가중치 분포를 2로 나누어 비 선형함수에서 쓰기 더욱 적합합니다. 
두 초기화 방법 모두 수렴 가능하지만 "He"초기화 방법이 더 빨리 수렴하는 것을 볼 수있습니다.
또한 오른쪽 그래프(30-layer모델)를 보면 "He"초기화 방법은 수렴을 하지만 "Xavier"초기화 방법은 학습을 완전히 지연시키고 gradient가 감소되는것을 관찰 할 수있습니다.
 
따라서 "He"초기화 방법이 더 깊은 모델에서 적용이 가능하다는 것을 알 수 있습니다.

<img src="/assets/images/posts/2019-11-27-He-initialization/table2.PNG" width="50%" alt="error"/>
하지만 ImageNet에서는 아직 큰 이점을 찾지 못했습니다.
30-layer 모델의 경우 38.56/16.59의 top-1/top-5 error를 가지는 반면에 위 표(14-layer)의 33.82/13.34 보다 훨씬 좋지 않음을 볼 수있는데
이는 layer가 깊을 수록 training error가 증가하기 때문이며 이 문제는 여전히 open problem 입니다.

결론적으로 "He"초기화 방법은 깊은 모델에서의 정확성에 대한 이점은 보여주지 못했지만 깊이 증가에 대한 더 많은 연구를 위한 토대를 마련했습니다.

### Experiments on ImageNet
<b>ReLU VS PReLU</b>
<img src="/assets/images/posts/2019-11-27-He-initialization/Fig4.PNG" width="50%" alt="error"/>
여기서 PReLU는 channel-wise 버전을 사용했고, ReLU와 PReLU 모두 같은 epoch으로 train하였습니다.
위 그래프는 training 동안 train/val error를 나타냈습니다.

PReLU는 ReLU에 비해 더 빨리 수렴되는 것을 볼 수있으며 PReLU의 train error와 val error 모두 ReLU보다 낮습니다.
따라서 PReLU가 ReLU에 비해 더 좋은 성능을 가지고 있음을 다시 한번 입증 하였습니다.

<b>Single-model Results and Multi-model Results</b>
<img src="/assets/images/posts/2019-11-27-He-initialization/realsinglemodel.PNG" width="50%" alt="error"/> 
A+ReLU가 VGG-19에 보고된 7.1% single-model의 결과 보다 상당히 좋습니다.
이는 얕은 모델을 미리 train 하지 않고 end-to-end train을 했기 때문이라고 보고 있습니다.
<img src="/assets/images/posts/2019-11-27-He-initialization/multimodel.PNG" width="50%" alt="error"/>
또한 C+PReLU는 5.7%로 multi-model 보다 좋은 결과를 가졌고, model B와 modle C를 비교했을때 C가 더 나음을 볼수있습니다.
(model B는 model A에 비해 deep하고, model C는 wide한 model입니다.)

따라서 모델이 충분히 깊을때 폭이 정확도에 필수적인 요소인것을 알 수있습니다.

<b>Surpassing Human-Level Performance?</b>



ImageNet 데이터셋에서 인간 성능이 약 5.1% top-5 error 인데 비해 이 연구는 4.94%의 error 결과를 도출 했습니다. 이는 인간 수준의 성과를 초과했음을 의미합니다.
<img src="/assets/images/posts/2019-11-27-He-initialization/Fig5.PNG" width="50%" alt="error"/>

이들의 방법으로 위 그림을 coucal”, “komondor”, “yellow lady’s slipper” 라고 성공적으로 인식을 하는 반면에 인간은 개,새,꽃 이라고 단순하게 인식을 합니다.
이렇게 특정 데이터셋에서는 우수한 결과를 도출하지만  일반적인 객체 인식에서 문맥의 이해나 고도의 지식이 필요한 경우에는 실수를 저지르기때문에 machine vision이  human vision을 능가하는것은 아닙니다.
그럼에도 이 결과는 시각적 인식에서 인간 수준의 성능과 일치하는 machine algorithm의 잠재적 가능성을 보여줍니다.


