# Confusing Tensorflow

## Intro
tensorflow를 이용해 내 맘대로 layer를 쌓으려고 할 때 시작부터 헷갈리는 것들이 너무 많다.
예를 들어, 단순한 convolution 2d layer를 한 층 쌓으려고 해도, tf.nn.conv2d를 사용해야 하는지,
tf.layers.conv2d를 사용해야 하는지, 혹은 tf.keras.layers에서 Conv2d를 import해와서 사용해야
하는지 알 수 없다. 또한, 그래프를 생성할 때 각 노드들에 이름을 부여해야 하는데, 어떤 노드에
이름을 부여할 수 있는지, 얼마나 잘게 쪼개서 이름을 부여하는 것이 좋은지, variable_scope는
어떻게 사용해야 하는지, tf.layers.conv2d의 "name" argument는 무엇인지 등에 대한 의문이 생긴다.

## Layers

### tf.nn.conv2d [[doc]](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)  
- 가장 기본이 되는 conv2d layer. 간단히 하려면 그냥 이걸 쓰면 된다.
- ex) conv = tf.nn.conv2d()
```python
tf.nn.conv2d(
input,
filter,
strides,
padding,
use_cudnn_on_gpu=True,
data_format='NHWC',
dilations=[1, 1, 1, 1],
name=None
)
```
- ```input```: A Tensor. Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
- ```filter```: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
- ```strides```: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
- ```padding```: A string from: "SAME", "VALID". The type of padding algorithm to use.

### tf.layers.conv2d [[doc]](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
- tf.layers는 신경망 구성을 손쉽게 해주는 유틸리티 모음이라고 한다(골빈해커).
- tf.nn.conv2d를 backend로 사용하기 때문에 작동원리는 사실상 같지만, 기능이 추가되어 있고, parameter가 조금 다르다.
- ex) conv = tf.layers.conv2d()
```python
tf.layers.conv2d(
inputs,
filters,
kernel_size,
strides=(1, 1),
padding='valid',
data_format='channels_last',
dilation_rate=(1, 1),
activation=None,
use_bias=True,
kernel_initializer=None,
bias_initializer=tf.zeros_initializer(),
kernel_regularizer=None,
bias_regularizer=None,
activity_regularizer=None,
kernel_constraint=None,
bias_constraint=None,
trainable=True,
name=None,
reuse=None
)
```
- ```inputs```: Tensor input.
- ```filters```: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
- ```kernel_size```: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
- ```strides```: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
- ```padding```: One of "valid" or "same" (case-insensitive).

### Conv2d(from tf.keras.layers)
- keras껀데, keras에서 넘어온 게 아니라면 일단 지금은 쓸 필요 없을 듯 하다.
그치만 keras API 쩐다고 하니, 알아두면 좋을 것이다.


## Scopes

### tf.name_scope()

### tf.variable_scope()

### tf.get_variable()


## Contrib
- contrib module containing volatile or experimental code.
- 텐서플로우 오픈소스 커뮤니티에서 개발에 기여한 코드를 반영했으나, 아직 테스트가 필요한 애들 모아놓은 모듈.

## Batch Normalization

### tf.layers.batch_normalization
- 이것을 사용하라!!
### tf.contrib.layers.batch_norm
 

## References
https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/how_tos/variable_scope/