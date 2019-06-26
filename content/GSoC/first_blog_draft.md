Title: Unifying Reifying and Symbolic-PyMC
Author: Joseph Willard
Date: 2019-6-10

# Introduction

Digging through TensorFlow I started by computing basic examples and
comparing them to numpy's output. While doing this I came across the
common theme of numerical approximation that theoretically should not
have been present. Of course this brought me to pondering what would I
have to do to get around these numerical errors that arise in software
today?

In this article consider the situation were one passingly uses an SVD.

    1  import numpy as np
    2  
    3  X = np.random.normal(0, 1, (10, 10))
    4  S = X.T.dot(X)
    5  U, d, Vt = np.linalg.svd(S)
    6  _ = S - np.dot(U*d, Vt)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<tbody>
<tr>
<td class="org-right">7.10542736e-15</td>
<td class="org-right">-1.15463195e-14</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">1.24344979e-14</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">-6.66133815e-16</td>
<td class="org-right">1.19904087e-14</td>
<td class="org-right">-4.6629367e-15</td>
<td class="org-right">3.33066907e-16</td>
<td class="org-right">4.4408921e-15</td>
</tr>


<tr>
<td class="org-right">-1.28785871e-14</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">-5.32907052e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">-1.52655666e-14</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">8.8817842e-15</td>
<td class="org-right">-3.55271368e-15</td>
</tr>


<tr>
<td class="org-right">1.24344979e-14</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">8.43769499e-15</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">6.21724894e-15</td>
</tr>


<tr>
<td class="org-right">8.8817842e-15</td>
<td class="org-right">-1.64313008e-14</td>
<td class="org-right">7.10542736e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">-6.21724894e-15</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">-6.66133815e-15</td>
<td class="org-right">2.22044605e-16</td>
<td class="org-right">-2.44249065e-15</td>
</tr>


<tr>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">0.0</td>
<td class="org-right">1.44328993e-15</td>
<td class="org-right">-4.4408921e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">-7.42461648e-16</td>
<td class="org-right">-1.99840144e-15</td>
<td class="org-right">1.11022302e-15</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">-1.77635684e-15</td>
</tr>


<tr>
<td class="org-right">-2.06501483e-14</td>
<td class="org-right">1.66533454e-14</td>
<td class="org-right">1.59872116e-14</td>
<td class="org-right">-9.76996262e-15</td>
<td class="org-right">6.52256027e-16</td>
<td class="org-right">0.0</td>
<td class="org-right">-1.33226763e-14</td>
<td class="org-right">4.4408921e-15</td>
<td class="org-right">5.77315973e-15</td>
<td class="org-right">-7.10542736e-15</td>
</tr>


<tr>
<td class="org-right">-1.99840144e-14</td>
<td class="org-right">1.06026299e-14</td>
<td class="org-right">1.7985613e-14</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">3.99680289e-15</td>
<td class="org-right">-1.42108547e-14</td>
<td class="org-right">2.66453526e-15</td>
<td class="org-right">4.4408921e-15</td>
<td class="org-right">-1.27675648e-14</td>
</tr>


<tr>
<td class="org-right">5.55111512e-15</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">6.66133815e-16</td>
<td class="org-right">0.0</td>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-7.07767178e-16</td>
<td class="org-right">2.66453526e-15</td>
</tr>


<tr>
<td class="org-right">3.33066907e-14</td>
<td class="org-right">-2.39808173e-14</td>
<td class="org-right">-2.04281037e-14</td>
<td class="org-right">1.17683641e-14</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-3.99680289e-15</td>
<td class="org-right">2.66453526e-14</td>
<td class="org-right">-7.91033905e-15</td>
<td class="org-right">-1.0658141e-14</td>
<td class="org-right">1.37667655e-14</td>
</tr>


<tr>
<td class="org-right">2.23154828e-14</td>
<td class="org-right">-1.42108547e-14</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">1.02140518e-14</td>
<td class="org-right">1.33226763e-15</td>
<td class="org-right">0.0</td>
<td class="org-right">1.44051437e-14</td>
<td class="org-right">-5.32907052e-15</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">7.10542736e-15</td>
</tr>
</tbody>
</table>

Let's see if TensorFlow exhibits the same issue?

     1  """ Seeing if tensorflow has the same issue
     2  """
     3  import tensorflow as tf
     4  from tensorflow.python.framework.ops import disable_eager_execution
     5  
     6  
     7  tf.compat.v1.InteractiveSession()
     8  disable_eager_execution()
     9  
    10  tfp = tfp.distributions
    11  
    12  X = np.random.normal(0, 1, (10, 10))
    13  
    14  S = tf.matmul(X, X, transpose_a=True)
    15  
    16  d, U, V = tf.linalg.svd(S)
    17  
    18  D = tf.matmul(U, tf.matmul(tf.linalg.diag(d), V, adjoint_b=True))
    19  ans = S - D
    20  
    21  _ = ans.eval()

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<tbody>
<tr>
<td class="org-right">-3.01980663e-14</td>
<td class="org-right">-4.4408921e-15</td>
<td class="org-right">2.39808173e-14</td>
<td class="org-right">4.4408921e-15</td>
<td class="org-right">7.99360578e-15</td>
<td class="org-right">-2.7533531e-14</td>
<td class="org-right">1.37667655e-14</td>
<td class="org-right">-1.59872116e-14</td>
<td class="org-right">2.48689958e-14</td>
<td class="org-right">7.10542736e-15</td>
</tr>


<tr>
<td class="org-right">-5.99520433e-15</td>
<td class="org-right">-1.24344979e-14</td>
<td class="org-right">6.88338275e-15</td>
<td class="org-right">-1.24344979e-14</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-1.82076576e-14</td>
<td class="org-right">-1.66533454e-15</td>
<td class="org-right">-5.77315973e-15</td>
<td class="org-right">-3.99680289e-15</td>
<td class="org-right">-1.95399252e-14</td>
</tr>


<tr>
<td class="org-right">2.13162821e-14</td>
<td class="org-right">2.88657986e-15</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">-7.99360578e-15</td>
<td class="org-right">2.57571742e-14</td>
<td class="org-right">-1.02140518e-14</td>
<td class="org-right">5.88418203e-15</td>
<td class="org-right">-1.55431223e-14</td>
<td class="org-right">3.33066907e-16</td>
</tr>


<tr>
<td class="org-right">5.77315973e-15</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">2.22044605e-16</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">-1.94289029e-15</td>
<td class="org-right">-1.0658141e-14</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">4.99600361e-15</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">-2.13162821e-14</td>
</tr>


<tr>
<td class="org-right">7.77156117e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-4.4408921e-15</td>
<td class="org-right">-1.22124533e-15</td>
<td class="org-right">-7.99360578e-15</td>
<td class="org-right">1.46549439e-14</td>
<td class="org-right">-4.08006962e-15</td>
<td class="org-right">-1.99840144e-15</td>
<td class="org-right">-1.0658141e-14</td>
<td class="org-right">1.55431223e-15</td>
</tr>


<tr>
<td class="org-right">-2.84217094e-14</td>
<td class="org-right">-1.9095836e-14</td>
<td class="org-right">2.66453526e-14</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">1.66533454e-14</td>
<td class="org-right">-4.08562073e-14</td>
<td class="org-right">3.55271368e-15</td>
<td class="org-right">-7.77156117e-16</td>
<td class="org-right">3.01980663e-14</td>
<td class="org-right">-1.59872116e-14</td>
</tr>


<tr>
<td class="org-right">1.28785871e-14</td>
<td class="org-right">-1.88737914e-15</td>
<td class="org-right">-1.0658141e-14</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">-3.1918912e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-4.97379915e-14</td>
<td class="org-right">3.90798505e-14</td>
<td class="org-right">1.19904087e-14</td>
<td class="org-right">-3.55271368e-14</td>
</tr>


<tr>
<td class="org-right">-1.59872116e-14</td>
<td class="org-right">-5.32907052e-15</td>
<td class="org-right">9.43689571e-15</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">-6.66133815e-16</td>
<td class="org-right">-2.44249065e-15</td>
<td class="org-right">3.37507799e-14</td>
<td class="org-right">-2.48689958e-14</td>
<td class="org-right">-1.15463195e-14</td>
<td class="org-right">1.0658141e-14</td>
</tr>


<tr>
<td class="org-right">2.30926389e-14</td>
<td class="org-right">-7.77156117e-16</td>
<td class="org-right">-1.28785871e-14</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">2.57571742e-14</td>
<td class="org-right">8.43769499e-15</td>
<td class="org-right">-1.24344979e-14</td>
<td class="org-right">-3.55271368e-14</td>
<td class="org-right">1.15463195e-14</td>
</tr>


<tr>
<td class="org-right">4.88498131e-15</td>
<td class="org-right">-2.13162821e-14</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-1.86517468e-14</td>
<td class="org-right">3.77475828e-15</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">-3.73034936e-14</td>
<td class="org-right">1.59872116e-14</td>
<td class="org-right">1.50990331e-14</td>
<td class="org-right">-5.32907052e-14</td>
</tr>
</tbody>
</table>

In regards to theory this should have been 0, but due to rounding
errors mostly drawn from limitations of floats this is not the
case. Questions one might ask are "Is there a way around this?"
and "Why would one care?"

Answering the second question, one reason is of course optimizing
numerical deficiencies when possible. The other could be this idea of
automating "pen and paper" math. This would allow someone with a
domain specific skill set be it in probability, numerical analysis to
be able to automate their "tricks" and demystify more complex ideas in
their respective fields.

Moving back to the first question, one method is to think of the
process of doing SVD above as a graph of operations. In this graph
each node is the output of an operation which are represented as the
edge connecting nodes. What this would allow us to do is to traverse
the graph looking for operations that could be reduced or removed all
together.


## An analogy

Consider an analogy using strings,

    input_src_code = """
    u, d, v = svd(S)
    ans = S - matmul(u, d, v)
    """

Consider some `optimize` function, that does the following:

    output_src_code = str_optimize(input_src_code)

where

    assert output_src_code == """
    u, d, v = svd(S)
    ans = S - S
    """

In this we need to replace `"matmul(u, d, v)"` with `"S"`, but what
do we need in order to implement this? 

1.  Match the pattern for an SVD, e.g.
    
        import re
        res = re.search("([a-zA-Z]+), ([a-zA-Z]+), ([a-zA-Z]+) = svd\(([a-zA-Z]+)\)", input_src_code)
        U = res.group(1)
        D = res.group(2)
        V = res.group(3)
        S = res.group(4)

2.  If it matches, match and replace the "matmul", e.g. with
    
        optimized_code = input_src_code.replace("matmul({}, {}, {})".format(U, D, V), S)

Using this analogy how does this map back to the TensorFlow objects
that we'll be working with?


# Graph reconstruction through TensorFlow

To begin answering the first question, let's look at what our term
`ans` has as objects outside of the standard assigned objects. As a
side note, this [blog post](https://blog.jakuba.net/2017/05/30/Visualizing-TensorFlow-Graphs-in-Jupyter-Notebooks/) covers some of the same material.

    1  _ = [i for i in dir(ans) if not i.startswith('_')]

    ['OVERLOADABLE_OPERATORS',
     'consumers',
     'device',
     'dtype',
     'eval',
     'get_shape',
     'graph',
     'name',
     'op',
     'set_shape',
     'shape',
     'value_index']

One that immediately strikes some interest is `ans.op`. 

    1  _ = ans.op

    <tf.Operation 'sub' type=Sub>

A `tf.Operation` is a node in the graph corresponds to a
computation. Some of the properties included in `tf.Operation` are
`inputs` and `outputs`. These could be the arguments to the operation
and the outputs, which corresponds to "S" and "matmul(&#x2026;)" for inputs
and "ans" for outputs in `input_src_code`.

Using our analogy, the above TensorFlow operation is the subtraction
in the string `input_src_code`.

    1  _ = [ans.op.inputs._inputs, ans.op.outputs]

    [[<tf.Tensor 'MatMul:0' shape=(10, 10) dtype=float64>,
      <tf.Tensor 'MatMul_2:0' shape=(10, 10) dtype=float64>],
     [<tf.Tensor 'sub:0' shape=(10, 10) dtype=float64>]]

These look like references to the previous tensors that were
subtracted to create `ans`. Of course I can directly check this.

    1  _ = [ans.op.inputs._inputs[0] == S, ans.op.inputs._inputs[1] == D]

    [True, True]

Great! So as a quick recap I now have a way to take the result `ans`
and walk backwards to our original matrices. Is it possible to
determine what kind of operations are transpiring? Specifically, is it
possible to determine if there was an SVD operation? The quick answer
is "yes"! All I need to do is use the same methods I've used thus
far.

    1  _ = ans.op.inputs._inputs[1].op.inputs._inputs[0].op

    <tf.Operation 'Svd' type=Svd>

This is like the "svd(&#x2026;)" in our analogy, so the argument to this
"string operator" is `op.inputs`.

At this point it's clear there exists a way to move through operations
and get the corresponding inputs and outputs. How do we do this using TensorFlow? We
know we would need a way to traverse a TensorFlow graph and find patterns like we
did above, which is analogous to searching strings with `re.search` and
replacing with `str.replace`.

In later blog posts I'll dive into creating functions that parse this
graph and make the required replacements much like our string
analogy. This is one of the main goals of the `symbolic-pymc` package
I'll be working with during GSoC 2019.

