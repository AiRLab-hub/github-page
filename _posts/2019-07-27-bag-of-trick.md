---
layout: post
current: post
cover:  assets/images/posts/2019-07-27-bag-of-tricks/cover.png 
navigation: True
title: Bag of Tricks
date: 2019-07-26 20:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---


Bag of Tricks for Image Classification with Convolutional Neural Networks 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 **"Bag of Tricks for Image Classification with Convolutional Neural Networks"** 입니다. 

이 논문의 저자들은 Image Classification에서 모델 구조의 변경(vgg,resnet,densenet)은 집중적인 주목을 받고 많은 사람들이 중요하다고 생각하지만, 나머지 Data Augmentations 및 Optimization Methods 과 같은 학습 방법들은 주목을 받지 못함에 안타까워 하고 있습니다. 또한 해당 방법들은 논문에서 간단하게 언급되거나, 심지어 논문에는 없고 소스 코드에만 존재하는 트릭들도 있습니다. 이 논문의 저자들은 이러한 학습 방법들을 자세히 조사하고 평가합니다. 아래의 사진을 보시면 트릭들만 잘 사용해도 4%의 정확도를 올릴 수 있는 것을 보실 수 있습니다.

![cover](/assets/images/posts/2019-07-27-bag-of-tricks/cover.png)

### Baseline

우선 여러 트릭들을 비교 하려면 동일한 환경을 만들어야 합니다. 이 논문의 저자들은 ResNet50을 기본 구조로 하고 아래의 나열된 방법으로 전처리를 했습니다.

1.  Randomly sample an image and decode it into 32-bitfloating point raw pixel values in[0,255].
(사실 저희가 훈련시키는 대부분의 이미지가 float 32bit에 srgb형식이기 때문에 특별한 경우가 아니라면 그냥 기본 상태 입니다.)

2.  Randomly crop a rectangular region whose aspect ratiois randomly sampled in[3/4,4/3]and area randomly sampled in[8%,100%], then resize the cropped regioninto a 224-by-224 square image.
(aspect ratiois을 지정하고 랜덤하게 이미지를 [224,224] 형식으로 크롭합니다.)

3.  Flip horizontally with 0.5 probability.
(50% 확률로 이미를 반전 합니다.)

4.  Scale hue, saturation, and brightness with coefficients uniformly drawn from[0.6,1.4].
(색조, 채도 및 밝기를 균일하게 스케일합니다.)

5.  Add PCA noise with a coefficient sampled from a normal distribution N(0,0.1).
(PCA노이즈를 이미지에 줍니다.)

모든 Conv,fc 층에는 Xavier Initialization을 하였고, Batch Normalization의 감마와 베타는 1,0 그리고 Optimizer는 NAG를 선택했습니다.
pytorch에서는 SGD의 파라메터중 nesterov를 True로 해주시면 됩니다.

위에서 언급한 베이스 라인들은 너무 일반적인 내용이라 추가 설명은 하지 않겠습니다. 아래 보이는 사진은 베이스라인으로 구현한 Image Classification의 정확도와 각 논문에서 주장하는 정확도의 비교입니다.

![table1](/assets/images/posts/2019-07-27-bag-of-tricks/table1.png)

### Large-batch training

최근 GPU 기술이 좋아지면서 Large-batch training이 가능해 졌습니다. 하지만 배치 사이즈가 증가하면 수렴속도가 늦어진다고 알려져 있고, 그 뜻은 테스트에서 정확도가 떨어질 확률이 있다는 것과 같습니다. 이 단란에서는 이러한 문제점을 경험적으로 풀어냅니다.

#### Linear scaling learning rate.

배치사이즈를 증가시키면 러닝레이트 또한 선형적으로 증가시켜줘야 합니다. 256배치 사이즈의 기준이 되는 러닝레이트를 0.1로 설정하고, 0.1 x b/256 이라는 공식을 제안합니다.(배치사이즈를 증가시키면 당연히 표본이 많이 선택되기 때문에 분산을 줄어듭니다. 분산이 줄어들면 안정적 이기 때문에 러닝레이트를 높이 설정해도 됩니다.)

#### Learning rate warmup.

초기의 parameters은 랜덤하게 초기화되기 때문에 불안정합니다. 그렇기 때문에 초기에 러닝레이트를 0에서 점점 키워 나가면서 초반 불안정을 해소합니다.

![Figure1](/assets/images/posts/2019-07-27-bag-of-tricks/Figure1.png)

#### Zero 감마.

residual block이 존재하는 모델에서는 bn의 감마를 0으로 초기합니다.(residual block쪽에서만) 그렇다면 초기의 네트워크가 짧아져서 학습이 더욱 빠르다고 합니다. 개인적인 사견이지만 모든 레이어에 일관적으로 초기화 해주는 것은 매우 간단하지만 특정 레이어만 다른 방법으로 초기화 해주는것은 매우 귀찮습니다. 그리고 실험해 본 결과 정확도에서 향상도 없고 학습속도가 그렇게 빨라지는 것도 아니기 때문에 그냥 감마는 1 베타는 0으로 초기화 하는것을 추천 드립니다.

#### No bias decay.

bias 는 decay를 안해줘야 오버피팅을 방지한다고 합니다. bn도 마찬가지로 decay를 안해줘야 오버피팅을 방지하는 효과가 있다고 합니다. pytorch에서는 bn의 감마와 베타가 weight, bias라는 이름으로 설정되어 있기 때문에 코드안에서 뭔가 특별한 것을 해줘야 합니다. 코드는 아래의 사진의 첨부했습니다.

![Figure2](/assets/images/posts/2019-07-27-bag-of-tricks/Figure2.png)

### Low-precision training

GPU가 발전하면서 FP16의 연산도 이제는 가능합니다. 당연히 기존의 FP32보다 메모리 소모량도 적고 연산량도 적기 때문에 FP16에서 연산하고 다시 FP32로 옮겨오면 당연하게 학습속도가 빨라집니다.

![Figrue3](/assets/images/posts/2019-07-27-bag-of-tricks/Figrue3.png)

![Figrues4](/assets/images/posts/2019-07-27-bag-of-tricks/Figrues4.png)

### Model Tweaks

원래의 resnet의 구조는 아래의 그림과 같습니다.

![Figrus5](/assets/images/posts/2019-07-27-bag-of-tricks/Figrus5.png)

사실 레즈넷을 보면서 많은 사람들이 7x7 conv를 보면서 3x3 conv 3개로 대체할 수 있는데 왜 했지? 그런 생각을 한번쯤은 모두 해보셨을거라고 생각합니다. 논문 저자는 7x7 conv를 3x3 conv 3개로 대체 하였습니다. 그리고 이미지를 다운샘플링 하는 부분에서 1x1 conv의 stride가 2인데 그렇다면 짝수번째의 핏쳐들은 모두 드랍되는것인데 이 부분도 문제라고 다들 생각하셨을 겁니다.이 논문의 저자는 이 부분도 수정하였습니다. 아래 그림 c 가 7x7을 수정한 부분이고, 그림 d 가 stride 2 의 문제를 수정하고, avgpool을 추가한 구조 입니다. 그림 b는 기존의 제안되었던 stride문제를 해결한 구조입니다.

![Figrus6](/assets/images/posts/2019-07-27-bag-of-tricks/Figrus6.png)

![Figrus7](/assets/images/posts/2019-07-27-bag-of-tricks/Figrus7.png)

### Training Refinements

정확도를 높이기위한 부분입니다.

#### Cosine Learning Rate Decay

이 논문에서는 Cosine Learning Rate Decay를 상당히 어렵게 설명합니다. 그냥 직관적으로 생각할때 Cosine Learning Rate Decay 에폭이 늘어 날때마다 당연하게도 러닝레이트는 줄어들어야 하는것은 당연하고, 그것을 매 에폭마다 설정해 주는 것 입니다. 학습 초반에는 수렴하지 않은 상태이기 때문에 큰 러닝 레이트를 가지고 있는 것이 유리하기 때문에 Cosine Learning Rate Decay는 처음에는 에폭이 조금 줄어들다가 수렴하기 시작하면 에폭이 많이 줄어 듭니다(기울기로 생각해 볼때).

![Figrus8](/assets/images/posts/2019-07-27-bag-of-tricks/Figrus8.png)

#### Label Smoothing

라벨 스무딩 입니다. 라베을 원핫 백퍼로 정답인 것은 1 아닌것은 0으로 만드는 것이 보통인데, 0.9 ,0.1 이런식으로 스무딩 해주는 것 입니다.
이부분은 잘 이해하지 못했지만, 직관적으로 한 쪽으로 너무 쏠려서 오버피팅이 될 확률이 적어 질 것 이라고 생각하고있습니다. 정확하게 아시는 분은 댓글 부탁 드립니다.

![Figrues9](/assets/images/posts/2019-07-27-bag-of-tricks/Figrues9.png)

#### Knowledge Distillation

선생님이 제자를 가르치듯이, 복잡하고 큰 모델이, 작은 모델의 학습을 도와 더 적은 복잡도로 좋은 성능을 내는 방법입니다. 선생 모델로 resnet152를 사용하고 학생모델로 resnet50을 선택했습니다.

#### Mixup Training

데이터 어규멘테이션의 한 방법이라고 생각하시면 됩니다. 두개의 클래스를 적당한 비율로 섞고 확률을 (0.7, 0.3) 이런식으로 표현합니다. 오버피팅을 방지하는 효과가 있고 경험적으로 너무 많이 섞으면 정확도가 오히려 떨어졌으며, 세그멘테이션 테스크에서는 매우 안좋은 결과가 나왔습니다. 개인적인 의견으로는 사용하지 않으셔도 될 것 같고, 네이버에서 이번에 발표한 cutmix 논문을 한번 참고 하시는게 좋으실 것 같습니다.

### 후기

이 논문을 딥러닝 입문 초기에 읽었으면 더 좋았을 것 같다는 생각을 했습니다. 나중에 후배들이 생기면 이 논문을 제일 첫 번째로 권해주고 싶을 정도로, 딥러닝을 많이 해봐야 얻을 수 있는 직관들과 꿀팁들을 알려주는 논문이라 매우 좋았습니다. 논문 구현 코드는 제 깃허브[https://github.com/seominseok0429/pytorch-warmup-cosine-lr](https://github.com/seominseok0429/pytorch-warmup-cosine-lr)에 올려 두었습니다.
