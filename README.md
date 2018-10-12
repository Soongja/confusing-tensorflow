## Session

***

## Graphs

#### tf.get_default_Graph()

#### sess.graph()

***

## Ops

모두 sess.run을 통해 수행해야 하는 operation들이다.

#### init_op

- 통상적으로 이것 -> tf.global_variables_initializer()
- 하지만 tf.data.Dataset을 사용하는 경우 Iterator의 initializer일 수도 있다.

#### loss_op

- 당연히 가장 중요. tf.summary.scalar로 기록한다.

#### train_op

- tf.train.AdamOptimizer(learning_rate).minimize(loss_op, var_list=t_vars)

#### summary_op

- summary에 관한 op.

***

## Scopes

#### tf.name_scope()

- A context manager for use when defining a Python op.

#### tf.variable_scope()

- A context manager for defining ops that creates variables (layers).

#### tf.get_variable()

- tf.get_variable()은 해당 variable_scope 내에서 variable이 없으면 생성하고, 있으면 불러오는 놈.
- name_scope는 무시되기 때문에 속편하게 variable_scope만 사용하자.

***

## Contrib

#### tf.contrib

- contrib module containing volatile or experimental code.
- 텐서플로우 오픈소스 커뮤니티에서 개발에 기여한 코드를 반영했으나,
아직 테스트가 필요한 애들 모아놓은 모듈.

#### tf.contrib.slim

- Slim is an interface to contrib functions, examples and models.

***

## Layers

#### tf.nn.conv2d [[doc]](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)  
- 가장 기본이 되는 conv2d layer. 간단히 만들려면 그냥 이걸 쓰면 된다.
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

#### tf.layers.conv2d [[doc]](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
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

#### Conv2d(from tf.keras.layers)
- keras껀데, keras에서 넘어온 게 아니라면 일단 지금은 쓸 필요 없을 듯 하다.
그치만 keras API 쩐다고 하니, 알아두면 좋을 것이다.

***

## Batch Normalization

#### tf.nn.batch_normalization

#### tf.layers.batch_normalization

- parameter로 training, trainable 두 가지가 있어서 둘 모두를 조절해야 함.

#### tf.contrib.layers.batch_norm

- 편리.
 - updates_collections=None에 대한 설명.  
 "One can set updates_collections=None to force the updates in place,
 but that can have a speed penalty, especially in distributed settings."
 - scale=True에 대한 설명.  
 "Note that by default batch_norm() only centers, normalizes, and shifts the inputs;
 it does not scale them (i.e., γ is fixed to 1). This makes sense for layers with no activation
 function or with the ReLU activation function, since the next layer’s weights can take care
 of scaling, but for any other activation function, you should add "scale": True to bn_params."

***

## Softmax

#### tf.nn.sparse_softmax_cross_entropy_with_logits
- argument를 넣을 때 logits=, labels= 이렇게 name을 명시해줘야 한다.

***

## What are saved?

#### Summary

- 텐서보드에 찍어보는 놈들. histogram, scalar, image 세 종류가 있음.
- tf.summary.scalar("name", tensor)
- writer = tf.summary.FileWriter(log_dir, sess.graph)
- writer.add_summary(step_summary, global_step=counter)

#### Checkpoint

- 모델 체크포인트.
- saver = tf.train.Saver(max_to_keep=5)
- saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
- saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

#### Graph

- 안드로이드에 올리기 위한 pb파일 만들기 위해 저장.
- tf.train.write_graph(sess.graph_def, logdir=log_dir, name='full_graph.pb', as_text=False)

***

## Summaries

#### tf.summary.merge_all()

- 모든 summary 다 합쳐줘

#### tf.summary.merge

- 원하는 summary들만 합칠 때 사용(ex. GAN에서 G_loss, D_loss들끼리 합칠 때)

***

## Global Step

#### global_step = tf.Variable(0, trainable=False, name='global_step')

- 이걸 편하게 해주는게 아래꺼

#### tf.train.get_or_create_global_step(graph=None)

- Returns and create (if necessary) the global step tensor.
- **graph**: The graph in which to create the global step tensor. If missing, use default graph.

***

## tf.Dataset

#### make_one_shot_iterator()

- 가장 쉬운 놈. initialize 불가능하기 때문에 한바퀴 더 iterate하려면 iterator 또 만들어줘야 한다.

#### make_initializable_iterator()

- initialize가 가능하기 때문에 여러 epoch을 돌리기에 가장 적합.

#### tf.data.Iterator.from_structure

- reinitializable한 놈. train iterator와 test iterator 번갈아가며 initialize할 때 적합.

***

## Printing out tensors

- print(sess.run(tensor))

***

## References
https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/how_tos/variable_scope/