Title: Converting `PyMC4` to `Symbolic-PyMC`
Author: Joseph Willard
Date: 2019-7-8


# Closing Loose Ends

Picking up from my last blog we are now in the position to
use `kanren` and `Symbolic-PyMC` together to walk and replace sections
in our SVD graph problem.

     1  import tensorflow as tf
     2  
     3  import numpy as np
     4  
     5  from unification import var
     6  
     7  from kanren import run, eq, lall
     8  
     9  from symbolic_pymc.etuple import etuple, ExpressionTuple
    10  from symbolic_pymc.relations.graph import graph_applyo
    11  from symbolic_pymc.tensorflow.meta import mt
    12  from symbolic_pymc.tensorflow.printing import tf_dprint
    13  
    14  
    15  X = tf.convert_to_tensor(np.random.normal(0, 1, (10, 10)), name='X')
    16  S = tf.matmul(X, X, transpose_a=True)
    17  d, U, V = tf.linalg.svd(S)
    18  S_2 = tf.matmul(U, tf.matmul(tf.linalg.diag(d), V, adjoint_b=True))
    19  ans = S - S_2
    20  
    21  def svd_reduceo(expanded_term, reduced_term):
    22      S_lv = var()
    23      d_mt, U_mt, V_mt = mt.linalg.svd(S_lv, name=var())
    24  
    25      t1 = mt.matrixdiag(d_mt, name=var())
    26      t2 = mt.matmul(t1, V_mt, transpose_a=False, transpose_b=True, name=var())
    27      template_mt = mt.matmul(U_mt, t2, transpose_a=False, transpose_b=False, name=var())
    28  
    29      # This is a workaround to reference issue #47.
    30      d_mt.op.node_def.attr.clear()
    31      t1.op.node_def.attr.clear()
    32      t2.op.node_def.attr.clear()
    33      template_mt.op.node_def.attr.clear()
    34  
    35      return lall(eq(expanded_term, template_mt),
    36  		eq(reduced_term, S_lv))
    37  
    38  
    39  def simplify_graph(expanded_term):
    40      expanded_term = mt(expanded_term)
    41      reduced_term = var()
    42  
    43      graph_goal = graph_applyo(svd_reduceo, expanded_term, reduced_term)
    44      res = run(1, reduced_term, graph_goal)
    45      res_tf = res[0].eval_obj.reify()
    46      return res_tf
    47  
    48  tf_dprint(ans)
    49  tf_dprint(simplify_graph(ans))

    Tensor(Sub):0,	shape=[10, 10]	"sub:0"
    |  Op(Sub)	"sub"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"X:0"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"X:0"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_2:0"
    |  |  |  Op(MatMul)	"MatMul_2"
    |  |  |  |  Tensor(Svd):1,	shape=[10, 10]	"Svd:1"
    |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  ...
    |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_1:0"
    |  |  |  |  |  Op(MatMul)	"MatMul_1"
    |  |  |  |  |  |  Tensor(MatrixDiag):0,	shape=[10, 10]	"MatrixDiag:0"
    |  |  |  |  |  |  |  Op(MatrixDiag)	"MatrixDiag"
    |  |  |  |  |  |  |  |  Tensor(Svd):0,	shape=[10]	"Svd:0"
    |  |  |  |  |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  |  |  |  |  ...
    |  |  |  |  |  |  Tensor(Svd):2,	shape=[10, 10]	"Svd:2"
    |  |  |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  |  |  ...
    Tensor(Sub):0,	shape=[10, 10]	"sub_1:0"
    |  Op(Sub)	"sub_1"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"X:0"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"X:0"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  ...

We have now seen a way to move from `TensorFlow` to `Symbolic-PyMC`
and traverse a graph. How does this relate to `PyMC4`?


# A look into new pymc4 models

As of the date this blog has been posted `PyMC4` received a large
update introducing generative models. In previous iterations of
`PyMC4` conversion would have involve trying to pinpoint what `TensorFlow`
object represented the observations. Luckily, with the recent changes
this can be controlled by how the model is created relieving some of
the searching on `Symbolic-PyMCs` part.

Consider the following model,

     1  from symbolic_pymc.tensorflow.meta import mt
     2  
     3  from tensorflow.python.framework.ops import disable_eager_execution
     4  disable_eager_execution()
     5  
     6  import numpy as np
     7  
     8  import pymc4 as pm
     9  
    10  from pymc4 import distributions as dist
    11  
    12  @pm.model(keep_return=False)
    13  def nested_model(intercept, x_coeff, x):
    14      y = yield dist.Normal("y", mu=intercept + x_coeff.sample() * x, sigma=1.0)
    15      return y
    16  
    17  
    18  @pm.model
    19  def main_model():
    20      intercept = yield dist.Normal("intercept", mu=0, sigma=10)
    21      x = np.linspace(-5, 5, 100)
    22      x_coeff = dist.Normal("x_coeff", mu=0, sigma=5)
    23      result = yield nested_model(intercept, x_coeff, x)
    24      return result
    25  
    26  ret, state = pm.evaluate_model(main_model())
    27  _ = [ret, state]

    [<tf.Tensor 'y_3_1/sample/Reshape:0' shape=(100,) dtype=float32>,
     SamplingState(
        values: ['main_model/intercept', 'main_model/nested_model/y', 'main_model']
        distributions: ['Normal:main_model/intercept', 'Normal:main_model/nested_model/y']
        num_potentials=0
    )]

Since the output of models in `PyMC4` are `TensorFlow` objects, which
`Symbolic-PyMC` is already setup to deal with. This means one can convert
`PyMC4` models to `Symbolic-PyMC` meta objects trivially by

    1  ret_mt = mt(ret)
    2  _ = ret_mt

    TFlowMetaTensor(tf.float32, TFlowMetaOp(TFlowMetaOpDef(obj=name: "Reshape"
    i...f.Operation 'y_3_1/sample/Reshape' type=Reshape>), 0, TFlowMetaTensorShape(100,),, obj=TensorShape([100])), 'y_3_1/sample/Reshape:0', obj=<tf.Tensor 'y_3_1/sample/Reshape:0' shape=(100,) dtype=float32>)

To move in reverse we only have to call reify on the new object

    1  _ = ret_mt.reify()

    <tf.Tensor 'y_3_1/sample/Reshape:0' shape=(100,) dtype=float32>


# Moving forward

From this point there are a few topics that need to be tackled. The
first is how do we implement the conversion of `PyMC4` models into
`Symbolic-PyMC` models behind the scenes? One way would be to expand
on the dispatcher that already runs on `TensorFlow` objects to now
consider `PyMC4` models. Other questions that have come up while
digging into this is whether there exists a way to reconstruct a graph
when eager mode is enabled.