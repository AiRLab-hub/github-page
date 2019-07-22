---
layout: post
current: post
cover:  assets/images/posts/2019-07-22-mobilenetv2/cover.png
navigation: True
title: MobileNetV2
date: 2017-07-22 10:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---

MobileNetV2 : Inverted Residuals and Linear Bottlenecks 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 MobileNetV2 : Inverted Residuals and Linear Bottlenecks 입니다. 간단하게 한 줄로 이 논문을 소개하자면 모바일이나, 임베디드에서도 실시간을 작동할 수 있게 모델이 경량화 되면서도, 정확도 또한 많이 떨어지지 않게하여, 속도와 정확도 사이의 트레이드 오프 문제를 어느정도 해결한 네트워크 입니다.

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure4.png)

먼저 이 논문을 읽기전에 알아두면 좋은 Related Works는 아래 두 논문 입니다.
* Xception: Deep Learning with Depthwise Separable Convolutions(https://arxiv.org/abs/1610.02357)
* MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications(https://arxiv.org/abs/1704.04861)

개인적으로 이 논문을 읽으면서 기존에 존재했던 "Xception", "MobileNets" 과의 크게 다른점을 저는 느끼지 못했습니다(그렇기에 이 논문을 제대로 이해 하시려면 앞에서 언급한 두 논문을 읽어보시는걸 추천 드립니다). 이 논문은 저자들이 "Xception"에서 제안했던  Depthwise Separable Convolutions을 그대로 사용합니다. 또한 Depthwise Separable Convolutions이 사용하는 철학, 가설을 그대로 채택합니다. 의식의 흐름대로 읽다보면 Depthwise Separable Convolutions이 뭐지? 하는 질문이 당연히 드실꺼라고 생각합니다.

### Depthwise Separable Convolutions이란?

Depthwise Separable Convolutions을 아주 간단하게 요약하면 **Depthwise Convolutions + Pointwise Convolutions** 입니다.

Depthwise Separable Convolutions을 설명하기 전에 기존의 Convolutions을 생각해 봅시다.
![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure5.png)

위 그림에 보이시는 것 처럼 기존의 Convolutions은 채널과 과 필터가 동시에 고려되서 최종 아웃풋을 만듭니다. 하지만 이 논문의 저자는 cross-channels correlation(입력 채널들 사이의 유사도)과 spatial correlation(필터와 하나의 특정 채널 사이의 관계)이 완전하게 독립적이기 때문에 **채널과 필터를 따로 분리해서 학습**을 진행해도 문제가 없다고 주장합니다. 실제로 연상량을 계산해보면 Traditional convolutions은 **입력 이미지의 크기x입력 이미지의 채널x 커널사이즈 제곱x아웃풋채널** 이지만 Depthwise Separable Convolutions의 연산량은 **입력 이미지의 크기x입력 이미지의 채널x (커널사이즈 제곱+아웃풋채널)** 이기 때문에 **8~9배** 정도 연산량이 줄어듭니다.(커널사이즈는 3 이라고 가정합니다)

다음으로 이 논문에서 주장하는 Linear Bottlenecks 입니다.

### Linear Bottlenecks 이란?

지금부터 조금 어려운 이야기를 직관적으로 쉽게 풀이하겠습니다(논문에서도 직관이라는 단어를 많이 사용합니다). 우선 manifold라는 말을 알고 있으셔야 합니다.
manifold란 어떤 이미지의 차원들이 존재하는 공간이라고 생각하시면 됩니다. 이 논문에서는 Manifold의 가설을 언급합니다(It has been longassumed  that  manifolds  of  interest  in  neural  networkscould be embedded in low-dimensional subspaces.). manifold 가설은 고차원의 정보는 사실 저차원으로 표현 가능하다는 것입니다. 예를 들어서 설명하면, 실제 세상에 존재하는 모든 사물들은 3차원 이라고 이야기를 하지만 사람들은 실제로 사물을 구분할 때는 2차원 정보를 받아들여 사물을 구분할 수 있다는 것 입니다. 즉 고차원 정보는 사실 저차원 정보로도 충분히 구분 할 수 있다는 것 입니다.

지금까지 Manifold에 대하여 설명한 이유는 이 논문에서 Linear Bottlenecks을 만들때 1x1의 pointwise Convolutions을 하여 차원수를 줄이기 때문입니다. **Manifold의 가설 그대로 고차원의 채널은 사실 저차원의 채널로 표현할 수 있다** 라는 논리 전개 입니다.(채널을 과도하게 줄이면 안됩니다. 예를들어서 사람은 3차원의 정보를 2차원으로 구분하지만 1차원으로는 구분 못하는 것과 같습니다.)

Linear Bottlenecks에서 주장하는 또 다른 하나는 ReLU는 필연적으로 정보 손실을 야기하기 때문에 어떤 특별한 작용을 해줘서 그 정보손실을 방어해야 한다는 것 입니다. 이제 그 특별한 작용에 대하여 말씀드리겠습니다.

시작하기 전에 가장 간단히 한줄로 요약하면 **"채널수가 충분히 많으면 ReLU를 사용해도 중요 정보는 보존된다!"** 입니다. 이 문장을 계속 상기시키면서 글을 읽으시면 이해하시는데 도움이 되실 것 같습니다. 

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure8.png)

위에 보이시는 그림처럼 채널이 1인 데이터가 ReLU를 지나면 중요 정보가 삭제 될 수 도 있습니다.

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure9.png)

하지만 위에 보이시는 그림처럼 채널이 2 인 데이터가 ReLU를 지나면 중요 정보가 삭제 되더라도 다른 채널에서는 아직까지 존재할 가능성이 채널이 많으면 많을수록 높기 때문에 **채널이 많을때 ReLU를 사용하면 괜찮다는 것** 입니다.(어차피 나중에 전부 합쳐져서 예측하기 때문에)

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure6.png)

위에 보이시는 그림은 이 주장을 실험적으로 증명한 것 입니다. 차원을 2, 3, 5, 15, 30 을 각각쓰고 ReLU를 쓰고 원래대로 복원하였습니다. 그림에서 보이는 것 과 같이 차원이 적을때는 ReLU를 쓰면 정보가 손실되어 원본 영상을 복원할 수 없지만 차원을 충분히 늘리고 ReLU를 쓰면 15, 30 과 같이 잘 복원 할 수 있다는 것 입니다.

마지막으로 Linear Bottlenecks은 ReLU를 적용하지 않습니다. 위에서 말씀드린것과 같이 차원이 매우많이 축소된 상태이기 때문에 ReLU를 사용하면 정보손실이 있을 수도 있기 때문입니다.

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure3.png)

실험적으로 증명을 했는데, Linear Bottlenecks에 ReLU6을 썻을때와 안썻을때의 정확도의 차이 입니다.(ReLU6를 사용하는 이유는 연산량에 있어서 이득을 볼 수 있다고 알아보았는데 정확하진 않습니다. 혹시 정확한 이유를 아시는분은 댓글 부탁드립니다.) 또 shortcut 위치에 대한 실험도 있습니다.

### Inverted Residuals이란?

앞에서 설명드린 Depthwise Separable Convolutions과 Linear Bottlenecks을 결합하면 Inverted Residuals 입니다.

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure1.png)


기존의 Residuals을 거꾸로 뒤집은 모양이라 Inverted Residuals이라고 부르는것 같습니다. 앞에서 언급한 논리되로 ReLU를 사용해야 하기 때문에 채널을 확장(pointwise Convolutions)하고 Depthwise Convolutions을 진행합니다. 또 Linear Bottlenecks에서 대로 다시 채널수를 줄입니다.

![MobileNetV2](/assets/images/posts/2019-07-22-mobilenetv2/figure2.png)

stride가 1일때는 shortcut이 있지만 strdie가 2 일때는 shortcut 이 없습니다. 이유는 논문에서 설명하지 않고 있지만 이미지의 크기가 줄어들때 정보의 선형성이 보장되 않기 때문이라고 추측하고 있습니다. 

### Memory Efficient Inference

논문에서는 gpu에서 내부 메모리와 외부 메모리가 있기 때문에 내부로 올릴때의 크기과 나갈때의 크기만 중요하기때문에 메모리 스왑적인 부분에서 봤을때도 이 논문에서 제안한 Inverted Residuals구조가 효율적이라고 주장하고 있습니다.

### 후기
논문리뷰 끝입니다. 논문의 Conclusions은 개인적인 견해가 필요하고, 이 부분은 이 글을 읽고 있는 독자 여러분이 편견없이 논문을 읽으면 좋겠다고 생각하여 리뷰하지 않겠습니다. 

코드는 cifar10에 적용한 것이고 [https://github.com/seominseok0429/cifar10-mobilenetv2-pytorch](https://github.com/seominseok0429/cifar10-mobilenetv2-pytorch) 에 배포해 두었습니다.

끝까지 읽어 주셔서 감사합니다!
