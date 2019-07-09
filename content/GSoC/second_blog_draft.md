Title: Unifying, Reifying and Symbolic-PyMC Continued
Author: Joseph Willard
Date: 2019-06-24


# Graph Reconstruction Through TensorFlow Part 2

In the last blog post I focused on looking through TensorFlow objects
and what could be used within these to recreate the graph of
operations. Considering the analogy given in the first blog I should
have enough information now to recreate the `str_optimize` function
for TensorFlow.

     1  """ Seeing if tensorflow has the same issue
     2  """
     3  import numpy as np
     4  import tensorflow as tf
     5  from tensorflow.python.framework.ops import disable_eager_execution
     6  
     7  disable_eager_execution()
     8  
     9  X = np.random.normal(0, 1, (10, 10))
    10  
    11  S = tf.matmul(X, X, transpose_a=True)
    12  
    13  d, U, V = tf.linalg.svd(S)
    14  
    15  D = tf.matmul(U, tf.matmul(tf.linalg.diag(d), V, adjoint_b=True))
    16  ans = S - D

Using `symbolic-pymc` in particular the `tf_dprint` function we
can inspect the graph.

    1  from symbolic_pymc.tensorflow.printing import tf_dprint
    2  
    3  _ = tf_dprint(ans)

    Tensor(Sub):0,	shape=[10, 10]	"sub_1:0"
    |  Op(Sub)	"sub_1"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_3:0"
    |  |  |  Op(MatMul)	"MatMul_3"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/a:0"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/b:0"
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_5:0"
    |  |  |  Op(MatMul)	"MatMul_5"
    |  |  |  |  Tensor(Svd):1,	shape=[10, 10]	"Svd_1:1"
    |  |  |  |  |  Op(Svd)	"Svd_1"
    |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_3:0"
    |  |  |  |  |  |  |  Op(MatMul)	"MatMul_3"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/a:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/b:0"
    |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_4:0"
    |  |  |  |  |  Op(MatMul)	"MatMul_4"
    |  |  |  |  |  |  Tensor(MatrixDiag):0,	shape=[10, 10]	"MatrixDiag_1:0"
    |  |  |  |  |  |  |  Op(MatrixDiag)	"MatrixDiag_1"
    |  |  |  |  |  |  |  |  Tensor(Svd):0,	shape=[10]	"Svd_1:0"
    |  |  |  |  |  |  |  |  |  Op(Svd)	"Svd_1"
    |  |  |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_3:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(MatMul)	"MatMul_3"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/a:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/b:0"
    |  |  |  |  |  |  Tensor(Svd):2,	shape=[10, 10]	"Svd_1:2"
    |  |  |  |  |  |  |  Op(Svd)	"Svd_1"
    |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_3:0"
    |  |  |  |  |  |  |  |  |  Op(MatMul)	"MatMul_3"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/a:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul_3/b:0"

The output the top layer (furthest left) represents the subtraction
that took place. Each subsequent step right moves effectively one step
down in the list of operations until the original inputs are reached.

From this point the next step is to write a function that can replace
the below portion,

### Input Block
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_2:0"
    |  |  |  Op(MatMul)	"MatMul_2"
    |  |  |  |  Tensor(Svd):1,	shape=[10, 10]	"Svd:1"
    |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/a:0"
    |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/b:0"
    |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul_1:0"
    |  |  |  |  |  Op(MatMul)	"MatMul_1"
    |  |  |  |  |  |  Tensor(MatrixDiag):0,	shape=[10, 10]	"MatrixDiag:0"
    |  |  |  |  |  |  |  Op(MatrixDiag)	"MatrixDiag"
    |  |  |  |  |  |  |  |  Tensor(Svd):0,	shape=[10]	"Svd:0"
    |  |  |  |  |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  |  |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/a:0"
    |  |  |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/b:0"
    |  |  |  |  |  |  Tensor(Svd):2,	shape=[10, 10]	"Svd:2"
    |  |  |  |  |  |  |  Op(Svd)	"Svd"
    |  |  |  |  |  |  |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  |  |  |  |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/a:0"
    |  |  |  |  |  |  |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/b:0"

with the following:

### Output Block
    |  |  Tensor(MatMul):0,	shape=[10, 10]	"MatMul:0"
    |  |  |  Op(MatMul)	"MatMul"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/a:0"
    |  |  |  |  Tensor(Const):0,	shape=[10, 10]	"MatMul/b:0"

How do we match graphs like [Input Block](#input-block) such that we obtain we obtain
the argument of the "Svd" operator (i.e `S`, the [Output Block](#output-block))? 


# Unification and Reification

The idea behind unification is to make two terms equal by finding
substitutions for logic variables that would satisfy equality. A logic
variable is like an unknown term in algebra and substitutions are
simply a mapping between logic variables and values. Let's look at a
few quick examples where `x` is a logic variable,

    1  from unification import unify, reify, var
    2  
    3  x = var('x')
    4  _ = [unify((4, x), (4, 5), {}),
    5       unify(['t', x, 'est'], ['t', 'e', 'est'], {}),
    6       unify((4, x), (2, 5), {})]

    [{~x: 5}, {~x: 'e'}, False]

Reification is the opposite operation to unification. This implies that it takes a
variable and a substitution and returns a value that contains no
variables. Below is a quick example using Matt Rocklin's [unification](https://github.com/mrocklin/unification) library,

    1  from unification import unify, reify, var
    2  _ = [reify(["m", x, "s", "i", "c"], {x:'u'}),
    3       reify((4, x), {x: 5})]

    [['m', 'u', 's', 'i', 'c'], (4, 5)]

The concepts of "unification" and "reification" are important in term
rewriting, and what we have been discussing up to this point is term
rewriting!

Now, we want to unify [Input Block](#input-block) with another graph containing a
logic variable as the input for an "Svd". We can then use this logic
variable to reify.

How do we do this with a TensorFlow graph? Using the unification
 library we already have support for most basic builtin types such as
 "str", "tuple" and "list". However, unification can be extended
 further by modifying `_unify` and `_reify`. This extension is
 something that `Symbolic_PyMC` uses to manipulate TensorFlow graphs.

    1  from symbolic_pymc.tensorflow.meta import mt
    2  
    3  S_lv = var()
    4  d_mt, U_mt, V_mt = mt.linalg.svd(S, compute_uv=var(),
    5  			full_matrices=var(), name=var())
    6  
    7  template_mt = mt.matmul(U, mt.matmul(mt.matrixdiag(d, name=var()), V,
    8  					    transpose_a=False, transpose_b=True, name=var()),
    9  			transpose_a=False, transpose_b=False, name=var())

    1  D_mt = mt(D)
    2  s = unify(D_mt, template_mt, {})
    3  _ = s

    {~_27: tf.float64, ~_23: tf.float64, ~_19: tf.float64, ~_18: 'MatrixDiag_1', ~_20: TFlowMetaTensorShape([Dimension(10), Dimension(10)],, obj=TensorShape([10, 10])), ~_21: 'MatrixDiag_1:0', ~_22: 'MatMul_4', ~_24: TFlowMetaTensorShape([Dimension(10), Dimension(10)],, obj=TensorShape([10, 10])), ~_25: 'MatMul_4:0', ~_26: 'MatMul_5', ~_28: TFlowMetaTensorShape([Dimension(10), Dimension(10)],, obj=TensorShape([10, 10])), ~_29: 'MatMul_5:0'}

Reification in this case is straightforward.

    1  _ = reify(S_lv, s)

    ~_5

In our running example we would walk the graph i.e. `ans` in our
case. The output would be a new graph where [Input Block](#input-block) has been
replaced with `S_lv`. What can we use to implement walking through a
graph?

The concepts of unification and reification are encapsulated in the
language [miniKanren](http://minikanren.org/) as `eq` and `run` respectively. Luckily, miniKanren has a python
implementation! 

    1  from kanren import eq, run
    2  x = var()
    3  _ = run(1, x, eq((1, 2), (1, x)))

    (2,)

In later posts I'll go into exactly how `Symbolic-PyMc` uses
miniKanren while adding relations such as `graph_applyo` to walk and
replace sections.

