Title: General Updates
Author: Joseph Willard
Date: 2019-8-05

Since my last blog post I am transitioning into working more in
minikanren. In the meantime I have created a PR that adds the ability
to index graphs. This will make it easier moving forward
when the user needs to study the graph piece by piece. For example,

     1  import pymc4 as pm
     2  
     3  from pymc4 import distributions as dist
     4  
     5  import tensorflow as tf
     6  
     7  from tensorflow.python.eager.context import graph_mode  
     8  
     9  from symbolic_pymc.tensorflow.printing import tf_dprint
    10  
    11  
    12  @pm.model
    13  def transform_example():
    14      x = yield dist.Normal('x', mu=0, sigma=1)
    15      y = yield dist.Normal('y', mu=0, sigma=1)
    16      q = tf.realdiv(x, y)
    17      return q
    18  
    19  
    20  with graph_mode():
    21      model = transform_example()
    22      obs_graph, state = pm.evaluate_model(model)
    23  
    24  _ = tf_dprint(obs_graph)
    25  

    Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
    [GCC 7.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    Tensor(RealDiv):0,	shape=[]	"RealDiv:0"
    |  Op(RealDiv)	"RealDiv"
    |  |  Tensor(Reshape):0,	shape=[]	"Normal_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"Normal_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"Normal_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"Normal_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"Normal_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"Normal_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1]	"Normal_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"Normal_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"Normal_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"Normal_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(BroadcastArgs):0,	shape=[0]	"Normal_1/sample/BroadcastArgs:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(BroadcastArgs)	"Normal_1/sample/BroadcastArgs"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_1/sample/Shape_1:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"Normal_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"Normal_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_1/sample/Shape_2:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_1/sample/concat_1/axis:0"
    |  |  Tensor(Reshape):0,	shape=[]	"Normal_2_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"Normal_2_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_2_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"Normal_2_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_2_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"Normal_2_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_2_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"Normal_2_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_2_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"Normal_2_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1]	"Normal_2_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"Normal_2_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"Normal_2_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"Normal_2_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_2_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(BroadcastArgs):0,	shape=[0]	"Normal_2_1/sample/BroadcastArgs:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(BroadcastArgs)	"Normal_2_1/sample/BroadcastArgs"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_2_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_2_1/sample/Shape_1:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_2_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_2_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_2_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"Normal_2_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"Normal_2_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_2_1/sample/Shape_2:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_2_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_2_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_2_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_2_1/sample/concat_1/axis:0"
    python.el: native completion setup loaded

The above output can be tedious to read. With the changes I'm
implementing one could look at the graph in a reduced form, such as,

    1  _ = tf_dprint(obs_graph, depth_index=range(3, 6))

    |  |  |  Op(Reshape)	"Normal_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"Normal_1/sample/add"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_1/sample/concat_1"
    |  |  |  Op(Reshape)	"Normal_2_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_2_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"Normal_2_1/sample/add"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_2_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_2_1/sample/concat_1"

Next I plan on tackling issue #19, which involves graph
normalization and creating goals for tensorflow.

