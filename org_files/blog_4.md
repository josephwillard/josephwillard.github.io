Title: Converting `PyMC4` to `Symbolic-PyMC`Continued
Author: Joseph Willard
Date: 2019-7-23


# Second evaluation and moving forward

During this second stretch I was able to have another PR accepted. This
also began the phase of looking into how one might convert a `PyMC4` model to
`symbolic-pymc`. This process was the topic of my last blog where I
not only closed my discussion on my svd problem. I also showed that
converting `PyMC4` model to a `symbolic-pymc` meta object is a pretty
straight forward operation after the recent changes to `PyMC4`. 

    What about converting a `PyMC4` model to a `symbolic-pymc` meta
object making improvements and then converting it back? In the
following ratio of normals example I am not accounting for shape since
`PyMC4` does not directly account for it without using `sample` from
the underlying `tensorflowprobability` object.

     1  from pymc4 import distributions as dist
     2  
     3  from symbolic_pymc.tensorflow.meta import mt
     4  
     5  import tensorflow as tf
     6  
     7  import tensorflow_probability as tfp
     8  
     9  from tensorflow.python.eager.context import graph_mode  
    10  
    11  @pm.model
    12  def transform_example():
    13      x = yield dist.Normal('x', mu=0, sigma=1)
    14      y = yield dist.Normal('y', mu=0, sigma=1)
    15      q = tf.realdiv(x, y)
    16      return q
    17  
    18  
    19  with graph_mode():
    20      model = transform_example()
    21      obs_graph, state = pm.evaluate_model(model)
    22  
    23  _ = tf_dprint(obs_graph)
    24  

    Tensor(RealDiv):0,	shape=[]	"RealDiv_1:0"
    |  Op(RealDiv)	"RealDiv_1"
    |  |  Tensor(Reshape):0,	shape=[]	"Normal_3_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"Normal_3_1/sample/Reshape"
    |  |  |  |  Tensor(AddV2):0,	shape=[1]	"Normal_3_1/sample/add:0"
    |  |  |  |  |  Op(AddV2)	"Normal_3_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_3_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"Normal_3_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_3_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"Normal_3_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_3_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"Normal_3_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1]	"Normal_3_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"Normal_3_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"Normal_3_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"Normal_3_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_3_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(BroadcastArgs):0,	shape=[0]	"Normal_3_1/sample/BroadcastArgs:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(BroadcastArgs)	"Normal_3_1/sample/BroadcastArgs"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_3_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_3_1/sample/Shape_1:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_3_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_3_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_3_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"Normal_3_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"Normal_3_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_3_1/sample/Shape_2:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_3_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_3_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_3_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_3_1/sample/concat_1/axis:0"
    |  |  Tensor(Reshape):0,	shape=[]	"Normal_4_1/sample/Reshape:0"
    |  |  |  Op(Reshape)	"Normal_4_1/sample/Reshape"
    |  |  |  |  Tensor(AddV2):0,	shape=[1]	"Normal_4_1/sample/add:0"
    |  |  |  |  |  Op(AddV2)	"Normal_4_1/sample/add"
    |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_4_1/sample/mul:0"
    |  |  |  |  |  |  |  Op(Mul)	"Normal_4_1/sample/mul"
    |  |  |  |  |  |  |  |  Tensor(Add):0,	shape=[1]	"Normal_4_1/sample/random_normal:0"
    |  |  |  |  |  |  |  |  |  Op(Add)	"Normal_4_1/sample/random_normal"
    |  |  |  |  |  |  |  |  |  |  Tensor(Mul):0,	shape=[1]	"Normal_4_1/sample/random_normal/mul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(Mul)	"Normal_4_1/sample/random_normal/mul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(RandomStandardNormal):0,	shape=[1]	"Normal_4_1/sample/random_normal/RandomStandardNormal:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  Op(RandomStandardNormal)	"Normal_4_1/sample/random_normal/RandomStandardNormal"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(ConcatV2):0,	shape=[1]	"Normal_4_1/sample/concat:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(ConcatV2)	"Normal_4_1/sample/concat"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_4_1/sample/concat/values_0:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(BroadcastArgs):0,	shape=[0]	"Normal_4_1/sample/BroadcastArgs:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Op(BroadcastArgs)	"Normal_4_1/sample/BroadcastArgs"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_4_1/sample/Shape:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_4_1/sample/Shape_1:0"
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4_1/sample/concat/axis:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4_1/sample/random_normal/stddev:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4_1/sample/random_normal/mean:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4/scale:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4/loc:0"
    |  |  |  |  Tensor(ConcatV2):0,	shape=[0]	"Normal_4_1/sample/concat_1:0"
    |  |  |  |  |  Op(ConcatV2)	"Normal_4_1/sample/concat_1"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[0]	"Normal_4_1/sample/sample_shape:0"
    |  |  |  |  |  |  Tensor(StridedSlice):0,	shape=[0]	"Normal_4_1/sample/strided_slice:0"
    |  |  |  |  |  |  |  Op(StridedSlice)	"Normal_4_1/sample/strided_slice"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_4_1/sample/Shape_2:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_4_1/sample/strided_slice/stack:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_4_1/sample/strided_slice/stack_1:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[1]	"Normal_4_1/sample/strided_slice/stack_2:0"
    |  |  |  |  |  |  Tensor(Const):0,	shape=[]	"Normal_4_1/sample/concat_1/axis:0"

Theoretically the division of two normal distributions produces a
Cauchy distribution. Looking at the above graph it's clear that it
does not consider this reduction. This becomes a perfect situation for
`symbolic-pymc`! To do this we need to again construct a template to
unify against like in the svd example. 


# Converting PyMC4 model to symbolic-pymc

As mentioned in my last blog converting `PyMC4` objects to
`symbolic-pymc` objects is relatively simple,

    1  model_mt = mt(obs_graph)
    2  _ = mt(obs_graph)

    TFlowMetaTensor(tf.float32, TFlowMetaOp(TFlowMetaOpDef(obj=name: "RealDiv"
    i..._1', obj=<tf.Operation 'RealDiv_1' type=RealDiv>), 0, TFlowMetaTensorShape(,, obj=TensorShape([])), 'RealDiv_1:0', obj=<tf.Tensor 'RealDiv_1:0' shape=() dtype=float32>)


# manipulating the symbolic-pymc graph

To manipulate the graph we need to create a goal (`cauchy_reduceo`) That
first takes a term we want to manipulate and creates a template to
unify against (`Q_mt`). With this template we need to then unify it
against a template representing what we want to substitute in the
output graph (`cauchy_mt`).

     1  from kanren import lall, eq, run
     2  from unification import var
     3  from symbolic_pymc.relations.graph import graph_applyo
     4  from symbolic_pymc.etuple import ExpressionTuple
     5  from tensorflow_probability.python.internal import tensor_util
     6  
     7  def cauchy_reduceo(expanded_term, reduced_term):
     8      ''' Goal used for unification.
     9      '''
    10      X_mt = mt.reshape(tfp_normal(0, 1), shape=var(), name=var())
    11      Y_mt = mt.reshape(tfp_normal(0, 1), shape=var(), name=var())
    12      cauchy_mt = tfp_cauchy(0., 1.)
    13      Q_mt = mt.realdiv(X_mt, Y_mt, shape=var(), name=var())
    14      return lall(eq( Q_mt, expanded_term),
    15  	eq(reduced_term, cauchy_mt))
    16  
    17  
    18  def simplify_graph(expanded_term):
    19      ''' evaluates goal.
    20      '''
    21      with graph_mode():
    22  	expanded_term = mt(expanded_term)
    23  	reduced_term = var()
    24  	graph_goal = graph_applyo(cauchy_reduceo, expanded_term, reduced_term)
    25  	res = run(1, reduced_term, graph_goal)
    26  	res_tf = res[0].reify()
    27  	return res_tf
    28  
    29  
    30  def tfp_normal(loc, scale):
    31      '''Used to create template for unifying.
    32      '''
    33      sampled = var()
    34      return mt.add(mt.mul(sampled, scale, name=var()), loc, name=var())
    35  
    36  
    37  def tfp_cauchy(loc, scale):
    38      '''Used to create template for unifying.
    39      '''
    40      shape = var()
    41      probs = mt.randomuniform(shape=shape, minval=0., maxval=1.)
    42      return mt.add(loc,
    43  		  mt.mul(scale,
    44  			 mt.tan(mt.mul(np.pi, mt.sub(probs, .5, name=var())), 
    45  				name=var()), name=var()), name=var())
    46  simplify_graph(obs_graph)


	Tensor(Add):0,  shape=[1000]    "Add_169:0"
	|  Op(Add)  "Add_169"
	|  |  Tensor(Const):0,  shape=[]    "Add_169/x:0"
	|  |  Tensor(Mul):0,    shape=[1000]    "Mul_339:0"
	|  |  |  Op(Mul)    "Mul_339"
	|  |  |  |  Tensor(Const):0,    shape=[]    "Mul_339/x:0"
	|  |  |  |  Tensor(Tan):0,  shape=[1000]    "Tan_169:0"
	|  |  |  |  |  Op(Tan)  "Tan_169"
	|  |  |  |  |  |  Tensor(Mul):0,    shape=[1000]    "Mul_338:0"
	|  |  |  |  |  |  |  Op(Mul)    "Mul_338"
	|  |  |  |  |  |  |  |  Tensor(Const):0,    shape=[]    "Mul_338/x:0"
	|  |  |  |  |  |  |  |  Tensor(Sub):0,  shape=[1000]    "Sub_169:0"
	|  |  |  |  |  |  |  |  |  Op(Sub)  "Sub_169"
	|  |  |  |  |  |  |  |  |  |  Tensor(RandomUniform):0,  shape=[1000]    "cauchy_169:0"
	|  |  |  |  |  |  |  |  |  |  |  Op(RandomUniform)  "cauchy_169"
	|  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,    shape=[1]   "cauchy_169/shape:0"
	|  |  |  |  |  |  |  |  |  |  Tensor(Const):0,  shape=[]    "Sub_169/y:0"

In the above code `tfp_normal` and `tfp_cauchy` are created to unify
against `tfp.distributions.normal` and `tfp.distributions.cauchy`. To
create these I looked at `tfp.distributions.normal._sample_n` and
`tfp.distributions.cauchy._sample_n` respectively.  Now would be a
good time to mention that `symbolic-pymc` no longer disables eager
mode by default. The way around this is with `tensorflow's` own
`graph_mode` as shown above in `simplify_graph`.

Another part of `symbolic-pymc` is it's access to most of
`tensorflow's` api. Using this api is as simple as calling
"mt.API\_NAME" for example `mt.add(1, 2)`. What this does in the
background is searches for the operation through
`op_def_library.OpDefLibrary` and returns the corresponding meta
object. It is important to use the "mt" representation because it
allows us to use logic variables; `var()` from the unification
library.


# Moving Forward

Now that it's possible to create a PyMC4 model, convert it to
symbolic\_pymc, manipulate the graph and then convert back to a usable
object I'll be focusing adding common algebraic operations.

