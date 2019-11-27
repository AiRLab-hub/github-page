---
layout: post
current: post
cover:  assets/images/posts/2019-11-22-manifold-mixup/img1.PNG
navigation: True
title: "Manifold Mixup: Better Representations by Interpolating Hidden States"
date: 2019-11-22 02:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---

### Manifold Mixup: Better Representations by Interpolating Hidden States 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 **"Manifold Mixup: Better Representations by Interpolating Hidden States"** 입니다. 

Manifold Mixup: Better Representations by Interpolating Hidden States은(이하 Manifold Mixup) 2019ICML에 통과된 논문으로 카테고리는 Classification, Data Augmentation 입니다. 또한 딥러닝의 대가 Yoshua Bengio가 저자로 참여된 논문입니다. Manifold Mixup은 Mixup 논문에서 영감을 얻었으며, 인풋으로 들어오는 이미지 뿐만아니라, hidden states사이에서도 mixup을 하자는게 주된 내용이고, CIFAR100에서는 mixup을 능가하는 성능을 보입니다.

### Introduction ###

Manifold Mixup을 이해하기 위해서는 우선 매니폴드 가 무엇인지 먼저 아셔야 합니다. 저는 이 논문을 이해할 정도로만 매니폴드를 설명할 것 이니, 혹시 더 궁금하시다면, <https://www.youtube.com/watch?v=o_peo6U7IRM&t=4692s> 이활석님의 매니폴드 설명을 참고하시면 됩니다.

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img2.PNG)

위의 그림이 매니폴드를 나타내는 그림 입니다. 매니폴드란 데이터가 사는 공간입니다. 위의 사진처럼 한 매니폴드가 있으면, 그 위에 모든 데이터를 표현할 수 있고, 그림처럼 똑같은 4 이더라도 다 다른 위치에 있습니다. 또한 사람눈으로는 매우 닮아있는 4 라고 할지라도 매니폴드상 거리가 멀수도, 가까울 수도 있습니다. 직관적으로 딥러닝은 이러한 매니폴드에서 공통된 특성을 가로지르는 하나의 선을 찾는 것 이라고 생각하셔도 될 것 같습니다.
이제 매니폴드를 간략하게 알았으니 다음 설명으로 넘어가겠습니다.

논문 저자는 친절하게 Manifold Mixup을 한줄로 요약해 줍니다. "Manifold Mixup improves the hidden representations and decision boundaries of neural networks at multiple layers." 즉 매니폴드 믹스업이란, "다중 레이어가 있을때 그냥 믹스업 처럼 인풋만 이미지를 섞어버리면 불공평하니, 다중 레이어 모든 핏쳐맵에서 섞자!" 입니다. 실제로 이 말이 이 논문의 전부이며 앞으로는 이 말을 증명하고 실험하는 과정입니다. 또한 이 논문은 라벨 스무딩의 효과가 있다. 라고 생각하시면 이해하기 편하실것 같습니다.

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img3.PNG)
![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img4.PNG)

위의 그림들은 저자가 2D spiral dataset으로 시각화한 그림입니다. 상단의 그림의 왼쪽은 아무것도 사용하지 않은 base이며, 상단의 오른쪽은 Manifold Mixup 입니다. 그림에서 보이는것과 같이 두개의 정확도는 비슷하지만, base는 오버피팅이 생긴 것을 한 눈에 알수 있고, Manifold Mixup은 오버피팅이 일어나지 않은 것을 볼수 있습니다. 또한 아래의 그림은 기존 유명 regularizers 들과의 성능을 정성적으로 비교한 것이며, 정성적으로는 Manifold Mixup이 좋아 보이나, "Manifold Mixup이 다른 타 regularizers들을 능가하는 방법이다!" 라고 생각하지 마시고, 그냥 저런 유명 방법들과 견줄만한 방법이다. 정도로만 생각하시면 좋을것 같습니다.

### Manifold Mixup ###

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img5.PNG)
![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img6.PNG)

위의 수식들은 Manifold Mixup을 이해하기 쉽게 저자가 수식으로 표현한 것이며, 상단의 수식은 매니폴드에 존재하는 hidden states를 섞고, label도 그 수치만큼 섞겠다는 이야기 입니다. 아래의 수식은 전체 학습 프로세스가 어떻게 작동되는지 설명한 수식이며, 학습이 진행되면 input을 포함한 hidden states에 Mixup이 동작한다는 수식입니다. 또한 Beta는 Beta분포를 따르는 것을 의미합니다. Beta분포를 사용한 이유는 랜덤하게 뽑으면 섞이는 두 대상이 일정하게 섞일 확률이 높기 때문에, 한쪽이 더 우세하게 섞기 위하여 Beta분포를 사용한 것 입니다.

### Empirical Investigation of Flattening ###

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img7.PNG)

논문 저자는 Flattening 하는 것이 왜 좋은가를 실험하기 위하여 MNIST데이터 셋에서 Singular Value Decomposition (SVD)을 통하여 실험합니다.
SVD를 직관적으로 설명하면 선형대수에서 배우는 특이값 분해로 같은 이미지에서 투영변환, 스케일변환, 회전변환을 하였을때 딥러닝 관점에서 augmentation이 된 데이터들 즉 매니폴드를 지나가는 직선의 거리 라고 생각하시면 될 것 같습니다. 이러한 SVD 값이 Maniflod Mixup이 가장 작았습니다. 위의 그림은 이것을 시각화 한 것이고, Manifold Mixup을 사용한 방법에서 MNIST 데이터들이 잘 분류된걸 확인할 수 있습니다.

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img8.PNG)

또한 논문저자는 위의 그래프와 같이 다양한 어규멘테이션에서 실험을 합니다. 논문저자가 주장한대로 Maniflod Mixup을하면 매니폴드를 직관하는 선을 찾기 쉬워져 같은 데이터라면 결과가 좋아야 합니다. 위 그래프도 논문저자가 주장한대로 매니폴드 믹스업은 다양한 데이터에서 강인합니다. 하지만 아쉬운 점은 이 논문 저자는 극한의 튜닝을 했습니다. epoch를 2000까지 돌리는둥 ,,, 최소 1200epoch를 학습합니다.

### Experiments ###

![Figure1](/assets/images/posts/2019-11-22-manifold-mixup/img9.PNG)

마지막으로 실험 입니다. 논문 저자는 CIFAR10, 100 TINY Imagenet에서 실험을 하였습니다. 위의 표는 그 결과 입니다. 보이시는것과 같이 CIFAR10,100에서는 베타분포를 만들때 사용하는 알파값이 2일때 성능이 가장 좋고 Manifold Mixup에서 성능이 가장 좋습니다. 하지만 TINY Imagenet에서는 mixup보다 성능이 낮은데, 아마 tiny라 그런것이고, full imagenet이면 manifold mixup이 더 좋습니다. 그 결과는 cutmix논문을 참고하시면 확인하실수 있습니다.

### 후기 ###

이 논문은 저의 직관과 비슷하여 재미있게 읽었던 논문 입니다. 하지만 epoch를 1500까지 맞춰줘야 논문 성능을 구현 할수 있는점(저는 1200에 구현했습니다 ㅎㅎ) 하이퍼파라메터를 공개하지 않은점. 수학적으로 너무 무거운점이 아쉬웠습니다.