---
layout: post
current: post
cover: "assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_1.png"
navigation: True
title: Exploring Randomly Wired Neural Networks for Image Recognition
date: 2019-11-26 10:00:00
tags: [paper-review]
class: post-template
subclass: 'post'
author: sangwoo
---

안녕하세요 AirLab 이상우입니다. 이번에 읽어본 논문은 Exploring Randomly Wired Neural Networks for Image Recognition 으로 Network Architecture 를 Random 하게 만들어보면 어떨까? 하는 생각을 가진 논문입니다.

<hr>

#### **<u>Introduction</u>** 

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_1.png" width='1000' hight='300'></center><br>

논문 저자는 네트워크의 정규한 패턴의 연결이 있는 Network Architecture가 꾸준히 발전해왔으며 이러한 연결들을 랜덤하게 연결하면 어떻게 되는지 실험을 해보았다고 합니다. 결과는 생각보다 놀라웠으며 기존의 Network 보다 성능이 더 좋거나 성능을 견줄만한 결과를 보여주었다고 합니다. 이로써 Network Architecture를 수동적으로 개발하는 것보다는 앞으로 Network generator 를 개발하는 것이 더 좋을 것이다라는 생각을 가지고 있습니다.<br>
자세하기 알아보기전에 이 논문은 랜덤하게 연결을 해보면 어떨까? 라는 것이 메인 아이디어인만큼 사실 인공지능에서 큰 영감을 줄만한 내용보다는 어떻게 하면 Network를 랜덤하게 연결하는지와 관련된 내용이 논문의 주된 구성입니다.
하지만 제 생각에는 이 논문이 Network Architecture 의 연구 방향을 바꿀만한 논문이라 다들 한번씩 읽어보시면 좋을거 같습니다. 이제 자세한 내용을 소개해드리겠습니다.<br>

#### **<u>Methodology</u>** 

이 논문에서 소개하는 Network generator 는  **g(Θ,s)** 라는 값을 가지고 네트워크를 생성합니다. 이 값이 가지는 의미를 하나하나 설명하겠습니다. <br>
<br>

**Θ** : 네트워크의 다양한 정보를 포함하고 있는 값입니다. 예를 들어 VGG generator 가 있다면 VGG-16 으로 만들건지 VGG-34로 만들지를 결정하고 Network 의 깊이,폭,필터의 크기 등을 지정할 수 있습니다.<br>

**g** : graph의 연결을 결정합니다. 예를 들면 ResNet generator 가 있다면 F(x)의 값을 연결을 g를 통해서 x+f(x) 를 만들어 주는 값입니다. <br>

**g(Θ)** : 집합 N을 반환합니다. 위에 값으로 생성된 연결과 설정을 가지고 만든 네트워크를 반환합니다.<br>

**s** : 이 과정을 몇번 반복할 것인지 정합니다. g(θ) 를 몇번 호출하여 랜덤 네트워크 패밀리를 구성할 수 있습니다.<br>

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_2.png" width='200' hight='70'></center><br>

논문 저자는 random graph 를 생성하고, 생성된 graph를 가지고 Network에 매핑을 시키는 방식을 가지고 만들었습니다. 그래서 Network generator 는 일반적인 graph를 생성하며 시작되고, 노드를 연결하는 일련의 노드와 edge를 생성합니다. edge는 위에 그림에서 보이는 노드로 들어오거나 나가는 화살표이며, 이는 데이더의 Flow라고 합니다. 파란색 원으로 구성된 부분은 노드라고 부르며 노드는 들어오는 데이터는 weight의 합계를 통해 conv로 들어가며 conv 는 ReLu - convolution - BN triplet 로 구성되어있다고 합니다. 또 노드는 몇개의 Input,Output edge를 가질 수 있다고 합니다. 이를 통해 graph 이론의 일반 graph 생성기를 자유롭게 사용하며 graph를 얻으면 신경망에 매핑이 된다고 합니다.

#### **<u>Random Graph Models</u>**

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_3.png" width='1000' hight='300'></center><br>

위에 보시다 싶이 3가지 방법으로 짜여진 graph들이 있습니다. 이는 위에서 설명한 일반 graph 생성기로 생성된 graph들이며 이 방법들이 어떻게 사용되었는지 설명을 해드리겠습니다.<br>
첫번째 방법으로는 Erdos-R ˝ enyi 으로 **ER**로 표시하고 있습니다. ER 은 N의 노드를 사용하는 경우, 임의의 두개의 노드는 다른 노드들과는 무관하게 edge가 P의 확률로 연결이 된다고 합니다. 이 방법은 모든 노드 쌍에 대해서 반복되며 ER은 P의 확률만을 가지고 있기때문에 ER(P) 로 표시한다고 합니다. <br>
두번째 방법으로는 Barabasi-Albert 으로 **BA**로 표시하고 있습니다. BA 은 순차적으로 새 노드를 추가하여 랜덤 graph를 생성하며 초기 상태는 edge가 없는 M노드부터 시작된다고 합니다. 이 방법은 순차적으로 M개의 edge가 있는 노드가 생성될 때까지 반복하며, 중복되는 방향의 edge는 생성하지 않는다고 합니다. 이 과정은 N개의 노드가 생길 때까지 반복하며, BA는 단일 파라미터 M을 가지며, BA(M)으로 표시됩니다. <br>
세번째 방법으로는 Watts-Strogatz 으로 **WS**로 표시하고 있습니다. WS 은 처음에 N노드는 정기적으로 링에 배치되고 각 노드는 인접한 K/2에 연결된다고 합니다. 그런 다음 시계방향 루프에서 모든 노드 V에 대해 시계방향 I번째 다음 노드에 연결하는 edge가 P의 확률로 연결이 됩니다. I는 1 ≤ i ≤ K/2 이며 K/2 번 반복된다고 합니다. <br>

#### **<u>Experiments</u>**

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_4.png" width='1000' hight='300'></center>
<center>figure1. Comparison on random graph generators : ER,BA, and WS</center><br>

일반 graph 생성기 3개로 생성된 네트워크들의 정확도를 비교한 사진입니다. 직관적으로 보이는 결과이니 자세한 설명은 생략하겠습니다. <br>

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_5.png" width='1000' hight='300'></center>
<center>figure2. ImageNet: small computation regime</center><br>

이 논문 저자가 생성한 랜덤 네트워크로 다른 논문의 네트워크들과 비교했을 때도 정확도 면에서 경쟁력이 있는 결과는 보여줍니다.

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_6.png" width='1000' hight='300'></center>
<center>figure3. ImageNet: large computation regime.</center><br>

이 결과는 FLOPs와 params 수가 현저히 적은데도 불구하고 다른 네트워크들과 비슷한 정확도를 보여주고 있습니다. 가장 정확도가 좋은 PNASNNet-5와 1.3% 가 나지만 FLOPs와 params 수가 확실히 차이가 나는것을 보실 수 있습니다.

<center><img src="/assets/images/posts/2019-11-26-Exploring-Randomly-Wired-Neural-Networks-for-Image-Recognition/figure_r_7.png" width='1000' hight='300'></center>
<center>figure4. COCO object detection</center><br>

COCO dataset 에서 backbone을 ResNet과 ResNext를 사용했을 때 보다 RandWire를 썻을 때 정확도가 전체적으로 향상되있는 것을 볼수 있습니다. FLOP은 비슷하거나 더 낮다고 하였습니다.<br>
논문을 마치며 저의 생각은 앞으로 네트워크 구조의 발전이 네트워크 생성기의 설계쪽으로 기울어져 갈것같습니다. 비록 논문의 내용이 인공지능에 영감을 줄만한 내용은 충분히 있었다고는 생각하지 않았으나 네트워크 구조에 대한 방향성이 바뀔 수 있는 논문이라 충분히 읽어볼만 하다고 생각합니다. 부족한 점은 저의 메일이나 댓글로 남겨주세요. 감사합니다.