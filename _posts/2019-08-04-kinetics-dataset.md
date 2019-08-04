---
layout: post
current: post
cover:  assets/images/posts/2019-08-04-kinetics-dataset/cover.PNG
navigation: True
title: Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
date: 2019-08-03 08:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: minseok
---


Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset 리뷰

안녕하세요. **AiRLab**(한밭대학교 인공지능 및 로보틱스 연구실) 서민석입니다. 제가 이번에 리뷰할 논문은 제목에도 써 있는것과 같이 **"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"** 입니다. 

이 논문은 딥마인드에서 발표한 kinetics dataset 논문입니다. 이 논문은 데이터셋 논문임에도 action recognition의 역사와 방향성을 제시해주기 때문에 action recognition에 입문하시는 분들이라면 꼭 읽어 보시는 걸 추천해 드립니다.

기존의 action recognition 문제에서 UCF101 과 HMDB51 같은 규모가 작은 데이터셋은 좋은 성능을 내기 어려웠습니다. 그래서 이 논문 저자는 ImageNet처럼 action recognition에도 빅 데이터셋이 필요성을 느끼고 Kinetics 데이터셋을 만듭니다. Kinetics 데이터셋은 400개의 클래스들 과  한 클래스당 400개가 넘는 clips가 존재하는 빅데이터 셋입니다.(현재는 클래스 700 버전도 업로드 되었습니다.) Kinetics 데이터셋을 학습시킨 파라메터로 전이학습을 진행하여 UCF101 과 HMDB51 과 같은 작은 규모의 데이터셋에서도 좋은 성능을 냈습니다. 또한 Two-Stream Inflated 3D ConvNet (I3D)을 제안하고 전이학습을 진행하여 HMDB51에서는 80.9% UCF101에서는 98.0%를 달성하였습니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/cover.PNG)

### The Old Ⅰ: ConvNet+LSTM

영상에서 25 프레임을 뽑아낸후, CNN을 돌려서 나온 결과를 LSTM으로 입력하여 sequential한 정보를 예측해 보겠다는 아이디어 입니다. 직관적으로도 배경을 제거하고 CNN에 들어가는 것이 아니기 때문에 액션보다는 배경에 큰 영향을 받고, 미세한 액션은 잘 찾지 못하는 단점이 있었습니다.(LSTM을 꼭 활용하고 싶으시다면 OCR처럼 액션을 하는 오브젝트를 디텍션한 후 크롭하는 방법을 추천 드립니다.)

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure1.PNG)

### The old Ⅱ: 3D ConvNets 

action recognition에 처음 입문하시는 분들이 가장 먼저 직관적으로 예측가능 한 방법 같습니다. 하지만 3D conv는 3D 컨브보다 더 많은 파라메터가 필요하며, 이는 학습을 어렵게 만듭니다. 그리고 여전히 배경에 영향을 많이 받기 때문에 작은 행동들은 많이 놓치는 경향이 보였습니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure2.PNG)

### Optical flow 란?

기존 영상처리에서 움직이는 객체를 추적할때 자주 사용하던 방법입니다. Optical flow를 사용하면 움직이는 객체의 x방향 y방향의 벡터를 뽑아 낼 수 있습니다. 이 논문에서는 TVL1방법을 사용 했습니다. TVL1 방법이란 두 프레임 사이의 변화한 점을 픽셀 단위로 추적하면서, 데이터 사이의 차이를 L1 norm으로 구하고, 전체 데이터의 분산을 사용하여 정규화 하는 방법입니다.(딥러닝을 사용하지 않고 Optical flow를 뽑는 방법중 가장 쓸만한 방법 입니다.)

```python
optical_flow = cv2.DualTVL1OpticalFlow_create()
flow = optical_flow.calc(prvs, next, None)
```

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure3.PNG)

### The old Ⅲ: Two-Stream Networks 

아직도 활발한 연구가 진행되고 있는 방법이지만, 이 논문 저자는 old 라고 표기했기 때문에 old 라고 표현하겠습니다! 이 방법은 RGB와 Optical flow를 사용한 2D conv 방법입니다. Optical flow를 사용하여 행동을 예측하기 때문에 action을 비교적 잘 찾지만, 아직도 여전히 찝찝한 부분은 남아있습니다. 2D conv이기 때문에 rgb 한장 optical flow한장 들어가기 때문에 한 영상에서 뽑은 프레임 사이의 관계를 예측하는 부분에서는 아직까지 뭔가 찝찝합니다. 또한 이 찝찝함을 해결하기 위하여 rgb와 optical flow에서 나온 결과를 concat하여 3D를 만든후 3D conv를 하는 방법도 있습니다.(이 방법들의 단점을 직관적으로 생각해 보면 rgb는 3D컨브를 해야 RGB프레임 간 관계를 잘 예측 할 수 있기 떄문에 이 방법들은 RGB간 관계를 알기 힘들어서 정확도가 낮게 나온다고 생각합니다. 혹시 이 부분이 틀리다면 댓글로 지적 부탁드립니다.)

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure4.PNG)

### The New: Two-Stream Inflated 3D ConvNets

RGB와 Optical flow를 동시에 활용한다는 면에서 Two-Stream 방법이고, 2D conv가 아니라 3D conv이기 떄문에 Two-Stream Inflated 3D ConvNets 이라고 정의하며, 앞에서 언급한 모든 방법들보다 이 논문의 실험에서는 정확도가 가장 높았습니다.
RGB를 3D conv 함으로써 시간정보를 계층적으로 만들 수 있지만, 그래도 여전히 action을 인식하기에는 부족한 면이 있기 때문에 Optical flow도 활용합니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure5.PNG)

###  Inflating 2D ConvNets into 3D

단순하게 2D conv를 3D conv로 변경하려면 시간축 디멘션 하나를 추가하고 pad를 줘서 shape를 맞춰주시면 됩니다. 논문에서 언급한 것 처럼 간단하게 N*N을 N*N*N을 만들어 주시면 됩니다.

### Bootstrapping 3D filters from 2D Filter

3D conv에서 ImageNet pre-trained 된 weight를 활용하려면 단순하게 weight를 N번 복제해 주시면 됩니다. 뭔가 직관적으로 하면 안될 것 같은 느낌이 들지만 이 논문에서는 실험적으로 이렇게 해서라도 ImageNet pre-trained된 weight를 사용하는게 좋다고 밝힙니다.


### Pacing receptive field growth in space, time and network depth

직관적으로 당연히 공간정보와 시간정보의 stride가 적절하게 조절되어야 합니다. 공간정보는 조금 변하는데 시간정보의 stride가 많이 변하면 공간정보를 제대로 못보고, 그렇다고 시간정보의 stride가 조금 변하면 그것은 정지영상과 다름이 없어 행동을 잘 인식하지 못하게 됩니다. 이 논문 저자는 Inflated Inception-V1에서 첫 번째와 두 번째의 Max-Pool레이어에서는 시간축의 stride의 크기를 1 로 설정하면 경험적으로 더 좋았다고 합니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure6.PNG)

### Two 3D Streams

RGB 정보만 이용해도 3D conv를 사용하면 시간정보를 볼 수 있지만, 이 논문에서는 그래도 Optical flow를 사용하면 경험적으로 더 정확도가 높았다고 합니다. 지금까지 내용을 간단하게 요약하면 아무리 3D conv를 사용하는게 좋고, 3D conv를 사용하더라도 Optical flow를 사용하는게 좋다 입니다.

### Conculusion

저는 원래 결론 쓰는 것을 좋아하진 않지만!(개인적인 견해가 들어갈 수 있기 때문에) 데이터셋 논문이기 때문에 결론을 작성하겠습니다. 논문에서 제시한 I3D방법을 사용하는 것이 정확도가 가장 높았고, 정확도 대비 파라메터가 그렇게 많지 않았습니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure7.PNG)

파라메터도 적은 편 입니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure8.PNG)

정확도도 i3d가 가장 높았으며 i3d 중에서도 optical flow도 활용하는 방법이 가장 좋았습니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure9.PNG)

앞에서도 언급했던 것 처럼 Imagenet에서 pre-trained된 것을 활용하는게 더 성능이 좋았습니다.

![Image](/assets/images/posts/2019-08-04-kinetics-dataset/figure10.PNG)

모든 방법에서 i3d 방법이 가장 좋았습니다.

### 후기

이 논문은 action recognition에 입문하는 사람이라면 꼭 읽어 보시는걸 추천드리고, 아직 코드 구현은 못했습니다. 구현이 완료 되는데로 링크 첨부 하겠습니다. 읽어주셔서 감사합니다.







