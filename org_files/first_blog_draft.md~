Title: Unifying Reifying and Symbolic-PyMC
Author: Joseph Willard
Date: 2019-10-06


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
<td class="org-right">1.0658141e-14</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">-2.77555756e-16</td>
<td class="org-right">-1.33226763e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">-5.99520433e-15</td>
<td class="org-right">-6.21724894e-15</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">-1.33226763e-15</td>
<td class="org-right">-1.55431223e-15</td>
</tr>


<tr>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">1.0658141e-14</td>
<td class="org-right">1.33226763e-15</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-1.88737914e-15</td>
<td class="org-right">-1.0658141e-14</td>
<td class="org-right">-6.21724894e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-3.66373598e-15</td>
<td class="org-right">-6.66133815e-15</td>
</tr>


<tr>
<td class="org-right">-1.72084569e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">-7.77156117e-16</td>
<td class="org-right">-2.88657986e-15</td>
<td class="org-right">5.55111512e-16</td>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">2.66453526e-15</td>
<td class="org-right">7.77156117e-16</td>
</tr>


<tr>
<td class="org-right">-4.88498131e-15</td>
<td class="org-right">2.22044605e-16</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">2.66453526e-15</td>
<td class="org-right">-7.21644966e-16</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">1.33226763e-15</td>
<td class="org-right">-1.33226763e-15</td>
</tr>


<tr>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">-7.77156117e-16</td>
<td class="org-right">-2.44249065e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">3.10862447e-15</td>
<td class="org-right">1.33226763e-15</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">-1.94289029e-15</td>
<td class="org-right">-8.8817842e-16</td>
</tr>


<tr>
<td class="org-right">-3.77475828e-15</td>
<td class="org-right">-5.32907052e-15</td>
<td class="org-right">-2.55351296e-15</td>
<td class="org-right">-2.44249065e-15</td>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">3.55271368e-15</td>
<td class="org-right">8.43769499e-15</td>
<td class="org-right">1.99840144e-15</td>
<td class="org-right">1.44328993e-15</td>
<td class="org-right">2.66453526e-15</td>
</tr>


<tr>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-9.76996262e-15</td>
<td class="org-right">3.33066907e-16</td>
<td class="org-right">3.05311332e-15</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">0.0</td>
<td class="org-right">3.55271368e-15</td>
</tr>


<tr>
<td class="org-right">4.4408921e-15</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-1.33226763e-15</td>
<td class="org-right">-2.22044605e-16</td>
<td class="org-right">-1.55431223e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">4.4408921e-16</td>
</tr>


<tr>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">2.22044605e-16</td>
<td class="org-right">0.0</td>
<td class="org-right">-2.22044605e-16</td>
<td class="org-right">6.66133815e-16</td>
<td class="org-right">1.33226763e-15</td>
<td class="org-right">-1.33226763e-15</td>
<td class="org-right">-4.4408921e-15</td>
<td class="org-right">-1.55431223e-15</td>
</tr>


<tr>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">-2.44249065e-15</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">3.55271368e-15</td>
<td class="org-right">0.0</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">-1.55431223e-15</td>
<td class="org-right">-8.8817842e-16</td>
</tr>
</tbody>
</table>

Let's see if TensorFlow exhibits the same issue?

     1  """ Seeing if tensorflow has the same issue
     2  """
     3  import tensorflow as tf
     4  import tensorflow_probability as tfp
     5  from tensorflow.python.framework.ops import disable_eager_execution
     6  
     7  
     8  tf.compat.v1.InteractiveSession()
     9  #disable_eager_execution()
    10  
    11  tfp = tfp.distributions
    12  
    13  X = np.random.normal(0, 1, (10, 10))
    14  
    15  S = tf.matmul(X, X, transpose_a=True)
    16  
    17  d, U, V = tf.linalg.svd(S)
    18  
    19  D = tf.matmul(U, tf.matmul(tf.linalg.diag(d), V, adjoint_b=True))
    20  ans = S - D
    21  
    22  _ = ans.eval()

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
<td class="org-right">-2.30926389e-14</td>
<td class="org-right">-2.22044605e-14</td>
<td class="org-right">-1.77635684e-15</td>
<td class="org-right">-1.88737914e-15</td>
<td class="org-right">7.10542736e-15</td>
<td class="org-right">-9.32587341e-15</td>
<td class="org-right">2.66453526e-15</td>
<td class="org-right">-9.76996262e-15</td>
<td class="org-right">-2.87270208e-15</td>
<td class="org-right">-2.22738494e-15</td>
</tr>


<tr>
<td class="org-right">-1.86517468e-14</td>
<td class="org-right">-3.90798505e-14</td>
<td class="org-right">6.66133815e-15</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">-1.22124533e-15</td>
<td class="org-right">-1.95399252e-14</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-3.8719028e-15</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">1.11022302e-15</td>
</tr>


<tr>
<td class="org-right">-3.10862447e-15</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">-2.48689958e-14</td>
<td class="org-right">1.77635684e-15</td>
<td class="org-right">1.19904087e-14</td>
<td class="org-right">4.6629367e-15</td>
<td class="org-right">6.21724894e-15</td>
<td class="org-right">5.32907052e-15</td>
<td class="org-right">2.39808173e-14</td>
<td class="org-right">3.10862447e-15</td>
</tr>


<tr>
<td class="org-right">-2.10942375e-15</td>
<td class="org-right">3.55271368e-15</td>
<td class="org-right">-2.22044605e-15</td>
<td class="org-right">-1.59872116e-14</td>
<td class="org-right">1.02140518e-14</td>
<td class="org-right">7.99360578e-15</td>
<td class="org-right">1.11022302e-15</td>
<td class="org-right">5.30825384e-15</td>
<td class="org-right">-1.11022302e-14</td>
<td class="org-right">-1.33226763e-15</td>
</tr>


<tr>
<td class="org-right">4.88498131e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">1.42108547e-14</td>
<td class="org-right">8.43769499e-15</td>
<td class="org-right">-2.13162821e-14</td>
<td class="org-right">-6.06459327e-15</td>
<td class="org-right">-4.88498131e-15</td>
<td class="org-right">-4.88498131e-15</td>
<td class="org-right">-7.10542736e-15</td>
<td class="org-right">1.99840144e-15</td>
</tr>


<tr>
<td class="org-right">-3.99680289e-15</td>
<td class="org-right">-1.59872116e-14</td>
<td class="org-right">9.32587341e-15</td>
<td class="org-right">8.8817842e-15</td>
<td class="org-right">-6.75848266e-15</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">-5.32907052e-15</td>
<td class="org-right">0.0</td>
<td class="org-right">6.66133815e-16</td>
</tr>


<tr>
<td class="org-right">4.4408921e-16</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">6.21724894e-15</td>
<td class="org-right">2.22044605e-15</td>
<td class="org-right">-5.77315973e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">-7.54951657e-15</td>
<td class="org-right">-1.60982339e-15</td>
</tr>


<tr>
<td class="org-right">-7.54951657e-15</td>
<td class="org-right">-2.51187959e-15</td>
<td class="org-right">8.8817842e-16</td>
<td class="org-right">5.64132074e-15</td>
<td class="org-right">-3.44169138e-15</td>
<td class="org-right">-2.66453526e-15</td>
<td class="org-right">1.44328993e-15</td>
<td class="org-right">-8.8817842e-15</td>
<td class="org-right">-6.21724894e-15</td>
<td class="org-right">-5.32907052e-15</td>
</tr>


<tr>
<td class="org-right">-7.99360578e-15</td>
<td class="org-right">-7.99360578e-15</td>
<td class="org-right">2.13162821e-14</td>
<td class="org-right">-1.11022302e-14</td>
<td class="org-right">-4.4408921e-16</td>
<td class="org-right">-4.4408921e-15</td>
<td class="org-right">-3.99680289e-15</td>
<td class="org-right">-1.24344979e-14</td>
<td class="org-right">-1.77635684e-14</td>
<td class="org-right">3.55271368e-15</td>
</tr>


<tr>
<td class="org-right">-3.21270788e-15</td>
<td class="org-right">-8.8817842e-16</td>
<td class="org-right">9.76996262e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">3.33066907e-15</td>
<td class="org-right">-2.44249065e-15</td>
<td class="org-right">-1.38777878e-15</td>
<td class="org-right">-3.55271368e-15</td>
<td class="org-right">0.0</td>
<td class="org-right">1.77635684e-15</td>
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


# What is a TensorFlow graph?

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

    <tf.Operation 'sub_4' type=Sub>

A `tf.Operation` is a node in the graph corresponds to a
computation. Some of the properties included in `tf.Operation` are
`inputs` and `outputs`. These could be the arguments to the operation
and the outputs, which corresponds to "S" and "matmul(&#x2026;)" for inputs
and "ans" for outputs in `input_src_code`.

Using our analogy, the above TensorFlow operation is the subtraction
in the string `input_src_code`.

    1  _ = [ans.op.inputs._inputs, ans.op.outputs]

    [[<tf.Tensor 'MatMul_12:0' shape=(10, 10) dtype=float64>,
      <tf.Tensor 'MatMul_14:0' shape=(10, 10) dtype=float64>],
     [<tf.Tensor 'sub_4:0' shape=(10, 10) dtype=float64>]]

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

    <tf.Operation 'Svd_4' type=Svd>

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

