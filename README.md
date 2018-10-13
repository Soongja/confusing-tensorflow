## Graph

(Class)tf.Graph

Graph는 Tensor와 Operation으로 구성된다!

#### Tensor

- (Class)tf.Tensor
- tf.constant, tf.Variable, tf.placeholder 등. Units of data that flow between Operations.

#### Operation

- (Class)tf.Operation
- Tensor를 input으로 받아 computation을 수행한다.

#### tf.get_default_graph() vs. sess.graph()

- Session에서는 non-default 그래프를 실행시킬 수 있다. 그렇기 때문에 두 개가 달라질 수는 있지만, 우리에게는 일어나지 않을 일이다. 코드들을 보면 두 개 중 아무거나 쓰는 것을 볼 수 있는데 tf.Session뒤에 argument가 없는 경우(대부분) 같다고 보면 된다.
- If no `graph` argument is specified when constructing the session, the default graph will be launched in the session

***

## Session

(Class)tf.Session

A class for running TensorFlow operations.

A `Session` object encapsulates the environment in which `Operation` objects are executed, and `Tensor` objects are evaluated.

#### Properties

- graph: 해당 session에서 launch된 graph
- graph_def: A serializable version of the underlying TensorFlow graph. 안드로이드에 올리기 위해 pb파일을 만들 때 필요하다.

#### Methods

- run: Runs operations and evaluates tensors in fetches.

```
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
```

- fetches에는 graph element(Operation, Tensor)를 담으면 되는데, single element여도 되고, graph elements를 담은 list, tuple, dict 등이어도 상관없다.

***

## Variable

(Class)tf.Variable

Variable이란 Operation을 실행함으로써 그 값이 바뀔 수 있는 Tensor이다.

#### tf.get_variable() and tf.variable_scope

- Variable을 만들고, 재사용하는 가장 좋은 방법은 tf.get_variable()을 사용하는 것이다. tf.get_variable()은 tf.variable_scope 내에서 해당 name의 Variable이 없으면 생성하고, 이미 존재하면 불러오는 놈이다. 따라서 우선 tf.variable_scope를 알아야 한다.
- tf.variable_scope는 Variable에 계층적으로, 그리고 이해하기 쉽게 name을 붙일 수 있도록 해주는 context manager이다(A context manager for defining ops that creates variables). 예를 들어, conv1이라는 variable_scope 아래에서 conv2d op을 정의하면 conv1/weights, conv1/biases라는 Variable이 생성된다.
- variable_scope를 사용하지 않고 Variable을 공유, 재사용할 수도 있지만 그렇게 하면 naming을 다 다르게 해야하고, 나중에 graph를 봤을 때 이해하기도 어렵다.

#### tf.name_scope vs. tf.variable_scope

- tf.name_scope는 Python op만을 정의할 때 사용하는 context manager이다(A context manager for use when defining a Python op).
- 따라서 tf.name_scope 아래에서 tf.get_variable을 사용하면 기존의 Variable을 불러올 수 없다.
- tf.variable_scope는 Variable과 Op 모두의 namespace를 만들어내므로 거의 tf.variable_scope만 사용하면 될 듯하다.

#### tf.global_variables_initializer()

- Variable은 꼭 initialize되어야 한다. 그렇지 않으면 Graph에서 해당 Variable value를 사용하는 Op이 실행되지 않는다. 따라서 꼭 sess.run(tf.global_variables_initializer())를 처음에 해주어야 한다. 

#### tf.global_variables(scope=None)

- scope를 정해주면 해당 variable_scope내의 Variable list를 return하고, 그렇지 않으면 그래프 내의 모든 Variable list를 return한다.
- tf.train.Saver()의 첫 번째 인자인 var_list에 넣어서 특정 Variable만 checkpoint를 저장할 때 사용할 수 있다. 하지만 scope=None인 경우에는 넣지 않아도 어차피 모든 Variable이 저장되니 넣어줄 필요가 없다.

***

## Ops

흔히들 이름짓고 sess.run을 통해 실행하는 ops를 정리해두었다.

#### init_op

- 통상적으로 이것이다. -> tf.global_variables_initializer()
- 하지만 tf.data.Dataset을 사용하는 경우 Iterator의 initializer일 수도 있다. 그런 경우 op 이름 다르게 하자.

#### loss_op

- 당연히 가장 중요하다. tf.summary.scalar로 기록한다.
- ex)tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=, labels=))

#### train_op

- ex)tf.train.AdamOptimizer(learning_rate).minimize(loss_op, var_list=t_vars)

#### summary_op

- 보통 tf.summary.merge() 혹은 tf.summary.merge_all()로 summary들을 합쳐 summary_op으로 사용한다. summary에 관한 내용은 따로 정리하였다.

***

## What are saved after training?

## Summary

텐서보드에서 시각화할 수 있는 놈들이다. histogram, scalar, image 세 종류가 있다.

보통 logs 디렉터리에 저장하며 콘솔 명령어 tensorboard --logdir=./logs로 확인할 수 있다.

#### example process

writer = tf.summary.FileWriter(log_dir, sess.graph)

tf.summary.image("pred_masks", self.preds, max_outputs=2)

- #을 2개 3개 4개로 쪼개서 분류 더 잘하자. max_outputs등에 대한 설명도 다 넣자.

writer.add_summary(step_summary, global_step=counter)

#### tf.summary.scalar

- scalar 값 기록. loss와 accuracy가 대표적.

#### tf.summary.image

- 이미지 기록. tensor의 dtype은 uint8 혹은 float32이어야 하고, [batch_size, height, width, channels]의 shape을 가지는데 channels는 1, 3, 4 중 하나여야 한다.

#### tf.summary.histogram

- data의 distribution을 파악할 수 있다.

#### tf.summary.merge()

- 원하는 summary들만 합칠 때 사용한다. ex)GAN에서 G_loss, D_loss들끼리 합칠 때

#### tf.summary.merge_all()

- 모든 summary 다 합쳐준다. summary 그룹화할 필요가 없는 경우에는 이걸 쓰면 된다.

## Checkpoint

- 학습된 모델 사용을 위한 체크포인트.
- saver = tf.train.Saver(max_to_keep=5)를 만들고, saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)으로 저장한다.
- saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

#### Graph

- 안드로이드에 올리기 위한 pb파일 만들기 위해 저장.
- tf.train.write_graph(sess.graph_def, logdir=log_dir, name='full_graph.pb', as_text=False)

***

## Contrib

#### tf.contrib

- contrib module containing volatile or experimental code.
- 텐서플로우 오픈소스 커뮤니티에서 개발에 기여한 코드를 반영했으나, 아직 테스트가 필요한 애들 모아놓은 모듈.

#### tf.contrib.slim

- Slim is an interface to contrib functions, examples and models.

***

## Layers

#### tf.nn.conv2d

- 가장 기본이 되는 conv2d layer. 간단히 만들려면 그냥 이걸 쓰면 된다.
- ex) conv = tf.nn.conv2d()

#### tf.layers.conv2d

- tf.layers는 신경망 구성을 손쉽게 해주는 유틸리티 모음이라고 한다(골빈해커).
- tf.nn.conv2d를 backend로 사용하기 때문에 작동원리는 사실상 같지만, 기능이 추가되어 있고, parameter가 조금 다르다.
- ex) conv = tf.layers.conv2d()

#### Conv2d(from tf.keras.layers)

- keras껀데, keras에서 넘어온 게 아니라면 일단 지금은 쓸 필요 없을 듯 하다.
그치만 keras API 좋다고 하니, 알아두면 좋을 것이다.

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
- logits의 dtype은 float16, float32, float64이어야 하고, labels의 dtype은 int32, in64이어야 한다. 

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