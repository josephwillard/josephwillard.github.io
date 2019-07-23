Title: Converting `PyMC4` to `Symbolic-PyMC` Continued
Author: Joseph Willard
Date: 2019-7-23


# Second evaluation and moving forward

During this second stretch I was able to have another PR accepted. I
also began looking into how one might convert a `PyMC4` model to
`symbolic-pymc`. This process was the topic of my last blog where I
not only closed my discussion on my svd problem. I also showed that
converting `PyMC4` model to a `symbolic-pymc` meta object is a pretty
straight forward operation after the recent changes to `PyMC4`. 

    What about converting a `PyMC4` model to a `symbolic-pymc` meta object
making improvements and then converting it back? Consider the following,

     1  import numpy as np
     2  
     3  import pandas as pd
     4  
     5  import pymc4 as pm
     6  
     7  from pymc4.distributions import abstract
     8  
     9  from pymc4 import distributions as dist
    10  
    11  from pymc4.distributions.tensorflow.distribution import BackendDistribution
    12  
    13  from unification import var
    14  
    15  from kanren import run
    16  
    17  from symbolic_pymc.tensorflow.meta import mt
    18  
    19  from symbolic_pymc.relations.tensorflow import *
    20  
    21  from symbolic_pymc.tensorflow.printing import tf_dprint
    22  
    23  import tensorflow as tf
    24  
    25  import tensorflow_probability as tfp
    26  
    27  from tensorflow.python.eager.context import graph_mode  
    28  
    29  @pm.model
    30  def transform_example():
    31      x = dist.Normal('x', mu=0, sigma=1).sample(shape=(1000, ))
    32      y = dist.Normal('y', mu=0, sigma=1e-20).sample(shape=(1000, ))
    33      q = x/y
    34      yield None
    35      return q
    36  
    37  
    38  with graph_mode():
    39      model = transform_example()
    40      obs_graph, state = pm.evaluate_model(model)
    41  
    42  _ = tf_dprint(obs_graph)
    
    Tensor(RealDiv):0,	shape=[1000]	"truediv_8:0"
    |  Op(RealDiv)	"truediv_8"
    |  |  Tensor(Reshape):0,	shape=[1000]	"x_9_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"x_9_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1000]	"x_9_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"x_9_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1000]	"x_9_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"x_9_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1000]	"x_9_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"x_9_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1000]	"x_9_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"x_9_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1000]	"x_9_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"x_9_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"x_9_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"x_9_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Identity):0,	shape=[0]	"x_9_1/sample/x_9/batch_shape_tensor/batch_shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(Identity)	"x_9_1/sample/x_9/batch_shape_tensor/batch_shape"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"x_9_1/sample/x_9/batch_shape_tensor/Const:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"x_9_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"x_9_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"x_9_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"x_9_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"x_9_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"x_9_1/sample/concat_1/axis:0"
    |  |  Tensor(Reshape):0,	shape=[1000]	"y_9_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"y_9_1/sample/Reshape"
    |  |  |  |  Tensor(Add):0,	shape=[1000]	"y_9_1/sample/add:0"
    |  |  |  |  |  Op(Add)	"y_9_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1000]	"y_9_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"y_9_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1000]	"y_9_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"y_9_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1000]	"y_9_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"y_9_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1000]	"y_9_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"y_9_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"y_9_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"y_9_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Identity):0,	shape=[0]	"y_9_1/sample/y_9/batch_shape_tensor/batch_shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(Identity)	"y_9_1/sample/y_9/batch_shape_tensor/batch_shape"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"y_9_1/sample/y_9/batch_shape_tensor/Const:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"y_9_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"y_9_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"y_9_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"y_9_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"y_9_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"y_9_1/sample/concat_1/axis:0"

Theoretically the division of two normal distributions produces a
Cauchy distribution. Looking at the above graph it's clear that it
does not consider this reduction. This becomes a perfect situation for
`symbolic-pymc`! To do this we need to again construct a template to
unify against like in the svd example. Now would be a good time to
mention that `symbolic-pymc` no longer disables eager mode by
default. The way around this is with `tensorflow's` own `graph_mode`
as shown above.


# Converting `PyMC4` model to `symbolic-pymc`

As mentioned in my last blog converting `PyMC4` objects to
~symbolic-pymc objects is relatively simple,

    1  model_mt = mt(obs_graph)
    2  _ = mt(obs_graph)

    TFlowMetaTensor(tf.float32, TFlowMetaOp(TFlowMetaOpDef(obj=name: "RealDiv"
    i..._8', obj=<tf.Operation 'truediv_8' type=RealDiv>), 0, TFlowMetaTensorShape(1000,),, obj=TensorShape([1000])), 'truediv_8:0', obj=<tf.Tensor 'truediv_8:0' shape=(1000,) dtype=float32>)


# manipulating the `symbolic-pymc` graph

This is where things become difficult and encapsulates my work moving
forward. To manipulate the graph we would do the following.

     1  from kanren import lall, eq, run
     2  from unification import var
     3  from symbolic_pymc.relations.graph import graph_applyo
     4  from symbolic_pymc.etuple import ExpressionTuple
     5  from tensorflow_probability.python.internal import tensor_util
     6  
     7  def cauchy_reduceo(expanded_term, reduced_term):
     8      X_mt = tfp_normal(0, 1)
     9      Y_mt = tfp_normal(0, 1)
    10      cauchy_mt = tfp_cauchy(0, 1)
    11      Q_mt = mt.realdiv(X_mt, Y_mt, name=var())
    12      return lall(eq(expanded_term, Q_mt),
    13  		eq(reduced_term, cauchy_mt))
    14  
    15  def simplify_graph(expanded_term):
    16      with graph_mode():
    17  	expanded_term = mt(expanded_term)
    18  	reduced_term = var()
    19  	graph_goal = graph_applyo(cauchy_reduceo, expanded_term, reduced_term)
    20  	res = run(1, reduced_term, graph_goal)
    21  	res_tf = res[0].eval_obj.reify()
    22  	return res_tf
    23  
    24  
    25  def tfp_normal(loc, scale, n=1000):
    26      # might need n (to track)
    27  
    28      sampled = mt.random.normal(
    29      shape=(1000, ), mean=0., stddev=1.)
    30      return mt.add(mt.mul(sampled, scale, name=var()), loc, name=var())
    31  
    32  
    33  def tfp_cauchy(loc, scale, n=1000):
    34  
    35      shape = mt.concat(0, [[n], batch_shape_tensor(loc, scale)])
    36      probs = mt.random.uniform(shape=shape.obj, minval=0., maxval=1.)
    37      return mt.add(float(loc),
    38  		  mt.mul(float(scale),
    39  			 mt.tan(mt.mul(np.pi, mt.sub(probs, .5, name=var())), 
    40  				name=var()), name=var()), name=var())
    41  
    42  
    43  def batch_shape_tensor(loc, scale):
    44    t = mt.broadcast_dynamic_shape(
    45        mt.shape(input=tensor_util.convert_immutable_to_tensor(loc)),
    46        mt.shape(input=tensor_util.convert_immutable_to_tensor(scale)))
    47    return t

    

One thing to point out is that `symbolic-pymc` has access to most of
`tensorflow's` api and using it is as simple as calling "mt.API\_NAME"
for example "mt.add(1, 2)". What this does in the background is
searches for the operation through `op_def_library.OpDefLibrary` and
returns to the user the corresponding meta object. 

We also need the "mt" representation because it allows us to use logic
variables; var() from the `unification` library; for example let's
look at the template that will be used for unification.

    1  with graph_mode():
    2      X_mt = mt.reshape(tfp_normal(0, 1), shape=(1000,), name=var())
    3      Y_mt = mt.reshape(tfp_normal(0, 1), shape=(1000,), name=var())
    4      Q_mt = mt.realdiv(X_mt, Y_mt, name=var())
    5  
    6  _ = tf_dprint(Q_mt)

    Tensor(RealDiv):0,	shape=Unknown	"~_6368"
    |  Op(RealDiv)	"~_6365"
    |  |  Tensor(Reshape):0,	shape=Unknown	"~_6352"
    |  |  |  Op(Reshape)	"~_6349"
    |  |  |  |  Tensor(Add):0,	shape=Unknown	"~_6348"
    |  |  |  |  |  Op(Add)	"~_6345"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=Unknown	"~_6344"
    |  |  |  |  |  |  |  Op(Mul)	"~_6341"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1000]	"random_normal_1353:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"random_normal_1353"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1000]	"random_normal_1353/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"random_normal_1353/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1000]	"random_normal_1353/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"random_normal_1353/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"random_normal_1353/shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"random_normal_1353/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"random_normal_1353/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"Const_17689:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"Const_17690:0"
    |  |  |  |  (TFlowMetaConstant(obj=<tf.Tensor 'Const_17691:0' shape=() dtype=int32>),)
    |  |  Tensor(Reshape):0,	shape=Unknown	"~_6364"
    |  |  |  Op(Reshape)	"~_6361"
    |  |  |  |  Tensor(Add):0,	shape=Unknown	"~_6360"
    |  |  |  |  |  Op(Add)	"~_6357"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=Unknown	"~_6356"
    |  |  |  |  |  |  |  Op(Mul)	"~_6353"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1000]	"random_normal_1354:0"
    |  |  |  |  |  |  |  |  |  ...
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"Const_17696:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=Unknown	"Const_17697:0"
    |  |  |  |  (TFlowMetaConstant(obj=<tf.Tensor 'Const_17698:0' shape=() dtype=int32>),)


# Next steps

Using the above template we need to match it to our model. Following
this we can replace it with a Cauchy representation and translate that
back for use. To properly unify it though we need to make certain
fields logic variables. This is where the next issue that needs to be
tackled starts. In particular, one of the objects that "mt" does not
properly use is `tf.random.normal`. I need this to work to apply the
correct logic variables to make unification possible. In the next few
weeks I'll be tackling this as well as adding basic algebra substitutions.

