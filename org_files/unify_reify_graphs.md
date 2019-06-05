Title: Unifying Reifying and Symbolic-PyMC
Author: Joseph Willard
Date: 


# Introduction

Digging through tensorflow we started by computing basic examples and
comparing them to numpy's output. While doing this we found that in
many cases there was a common theme of numerical approximation that
theoretically should not have been present. Of course this brought me
to pondering what would we have to do to get around these numerical
errors that arise in software today?

In this article I'll start by walking through the two examples and
show the numerical errors, then I'll muse on what could be done to get
around this and introduce minikanren.


# Motivation

To start Let's look at a basic example using SVD.

    1  import numpy as np
    2  
    3  X = np.random.normal(0, 1, (10, 10))
    4  S = X.T.dot(X)
    5  U, d, Vt = np.linalg.svd(S)
    6  print(S - np.dot(U*d, Vt))

    [[ 1.77635684e-15  0.00000000e+00 -3.33066907e-16  8.43769499e-15
      -1.77635684e-15  1.77635684e-15 -3.10862447e-15 -3.77475828e-15
      -9.99200722e-16 -2.77555756e-15]
     [-3.55271368e-15 -5.32907052e-15 -1.77635684e-15  1.13103971e-15
      -5.32907052e-15  4.99600361e-15 -8.88178420e-16 -8.88178420e-16
       3.33066907e-16  1.66533454e-15]
     [-5.55111512e-16 -3.55271368e-15 -3.55271368e-15  1.77635684e-15
      -3.49720253e-15 -7.77156117e-16  5.32907052e-15  0.00000000e+00
      -7.77156117e-16 -3.88578059e-15]
     [ 1.77635684e-15  4.53109772e-15  3.55271368e-15  0.00000000e+00
      -6.66133815e-16  2.66453526e-15 -7.10542736e-15 -4.44089210e-16
       3.10862447e-15 -4.44089210e-16]
     [-2.22044605e-15  2.66453526e-15 -6.38378239e-15 -8.88178420e-16
       1.77635684e-15 -8.88178420e-16  7.10542736e-15 -4.44089210e-16
       1.33226763e-15  6.66133815e-16]
     [-4.44089210e-16 -1.88737914e-15 -1.22124533e-15  1.77635684e-15
      -8.88178420e-16  4.44089210e-15 -5.66213743e-15 -4.44089210e-16
      -1.11022302e-15 -8.88178420e-16]
     [ 0.00000000e+00 -4.21884749e-15  6.21724894e-15 -7.10542736e-15
       1.77635684e-15 -2.44249065e-15 -8.88178420e-15  5.32907052e-15
      -2.66453526e-15  5.77315973e-15]
     [-9.99200722e-16  4.44089210e-15 -2.22044605e-15 -3.55271368e-15
       3.99680289e-15 -4.66293670e-15  8.88178420e-16  0.00000000e+00
      -3.10862447e-15  8.88178420e-16]
     [ 0.00000000e+00 -3.38618023e-15  5.55111512e-16  2.66453526e-15
       2.66453526e-15 -4.44089210e-16 -4.44089210e-15 -2.22044605e-15
      -5.32907052e-15 -1.60982339e-15]
     [-1.11022302e-16  1.55431223e-15 -1.22124533e-15  2.22044605e-15
       8.88178420e-16  1.99840144e-15  3.99680289e-15 -6.66133815e-16
       4.49640325e-15  0.00000000e+00]]

     1  """ Seeing if tensorflow has the same issue
     2  """
     3  import tensorflow as tf
     4  import tensorflow_probability as tfp
     5  from tensorflow.python.framework.ops import disable_eager_execution
     6  
     7  tf.InteractiveSession()
     8  disable_eager_execution()
     9  
    10  tfp = tfp.distributions
    11  X = tfp.Normal(loc=0, scale=1)
    12  X = X.sample([10, 10])
    13  
    14  S = tf.matmul(X, X, transpose_a=True)
    15  
    16  d, U, V = tf.linalg.svd(S)
    17  
    18  #ans = S - tf.tensordot(U*d, Vt, 1)
    19  ans = S - tf.matmul(U, tf.matmul(tf.linalg.diag(d), V, adjoint_b=True))
    20  print(ans.eval())
    21  # Chris was suggesting something like this to turn off eager mode
    22  # import tensorflow.compat.v2 as tf
    23  # tf.enable_v2_behavior

    /home/joseph/anaconda3/envs/symbolic-pymc/lib/python3.6/site-packages/tensorflow/python/client/session.py:1702: UserWarning: An interactive session is already active. This can cause out-of-memory errors in some cases. You must explicitly call `InteractiveSession.close()` to release resources held by the other session(s).
      warnings.warn('An interactive session is already active. This can '
    [[-3.81469727e-06  7.62939453e-06  5.48362732e-06  8.34465027e-07
      -1.37090683e-06  3.33786011e-06 -7.00354576e-07 -2.14576721e-06
       6.19888306e-06 -1.43051147e-06]
     [ 6.19888306e-06 -1.23977661e-05 -8.58306885e-06 -2.86102295e-06
       1.19209290e-06 -7.86781311e-06  1.71363354e-06  3.57627869e-06
      -1.09672546e-05  1.43051147e-06]
     [ 4.88758087e-06 -9.53674316e-06 -1.23977661e-05  8.41915607e-07
       3.57627869e-06 -3.33786011e-06 -2.62260437e-06  2.64495611e-06
      -4.76837158e-06  2.26497650e-06]
     [ 8.94069672e-07 -2.32458115e-06 -1.26659870e-07 -1.04904175e-05
      -9.53674316e-06 -5.12599945e-06 -7.15255737e-07  6.67572021e-06
      -4.17232513e-06 -2.62260437e-06]
     [-8.94069672e-07  9.53674316e-07  1.19209290e-06 -6.19888306e-06
      -2.09808350e-05 -1.33514404e-05  9.53674316e-06  1.19209290e-06
      -6.19888306e-06 -9.29832458e-06]
     [ 3.09944153e-06 -7.62939453e-06 -5.24520874e-06 -2.86102295e-06
      -1.14440918e-05 -2.57492065e-05  1.52587891e-05  1.66893005e-06
      -1.71661377e-05 -6.19888306e-06]
     [-7.89761543e-07  9.23871994e-07 -3.33786011e-06 -1.19209290e-06
       8.82148743e-06  1.66893005e-05 -1.43051147e-05  2.38418579e-06
       1.04904175e-05  4.52995300e-06]
     [-1.66893005e-06  2.14576721e-06  1.86264515e-06  3.81469727e-06
       7.15255737e-07  1.19209290e-06  1.90734863e-06 -2.86102295e-06
       4.05311584e-06  4.52995300e-06]
     [ 6.19888306e-06 -1.09672546e-05 -6.31809235e-06 -4.52995300e-06
      -6.07967377e-06 -1.76429749e-05  9.05990601e-06  5.24520874e-06
      -1.52587891e-05  0.00000000e+00]
     [ 0.00000000e+00  5.96046448e-07  6.37024641e-07 -2.38418579e-06
      -8.10623169e-06 -6.55651093e-06  3.09944153e-06  2.86102295e-06
      -9.53674316e-07 -4.76837158e-06]]

In regards to theory this should have been 0, but due to rounding
errors mostly drawn from limitations of floats this is not the case. A
natural question to ask is whether there is a way around this. To
provide an answer to this we need to introduce minikanren/logpy and
the concepts of unify, reify and goals.

Why would we care about this? One reason is of course numerical
accuracy. The other could be this idea of automating "pen and paper"
math. This would allow someone with a domain specific skill set be it
in probability, numerical analysis to be able to automate their
"tricks" and demystify.

How do I find when a SVD is being computed? How do I then work with it?


# Need to write down information that we need


### Talk about what comprises a TF object

There exists an op field, etc. Identify SVD operation


### A way to get parents


### What components made up the multiplication


### show method to parse graph of operations and replace SVD example using minikanren. (need to do this with expository dialogue)

     1  def svd_optimize(graph):
     2      # graph.op
     3      # graph.op.inputs
     4      # graph.op.outputs
     5      # graph.op.op_def
     6      # graph.op.get_value
     7      # walk through these with ans
     8      pass
     9  
    10  svd_optimize(ans)
    11  
    12  # This function produces a graph
    13  tf.linalg.svd


# Unify

The idea behind unify is to take two similar terms and form a **substitution** which can be thought of as a mapping between variables and values. Let's look at a few quick examples,

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Constant</td>
<td class="org-left">Variable</td>
<td class="org-left">Substitution</td>
</tr>


<tr>
<td class="org-left">(4, 5)</td>
<td class="org-left">(x, 5)</td>
<td class="org-left">{x: 4}</td>
</tr>


<tr>
<td class="org-left">'test'</td>
<td class="org-left">'txst'</td>
<td class="org-left">{x: 'e'}</td>
</tr>
</tbody>
</table>

In layman's terms at this point we are looking for effectively the set of values that make the statement true. Below are some examples of terms that do not unify,

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Constant</td>
<td class="org-left">Variable</td>
<td class="org-left">Substitution</td>
</tr>


<tr>
<td class="org-left">(4, 5)</td>
<td class="org-left">(3, x)</td>
<td class="org-left">NA</td>
</tr>


<tr>
<td class="org-left">'test'</td>
<td class="org-left">'exror'</td>
<td class="org-left">NA</td>
</tr>
</tbody>
</table>


# Reify

Reify is the opposite operation to unify. This implies that it takes a variable and a substitution and returns a value that contains no variables. Below is a quick example,

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Variable</td>
<td class="org-left">Substitution</td>
<td class="org-left">Constant</td>
</tr>


<tr>
<td class="org-left">(x, 10)</td>
<td class="org-left">{x: 5}</td>
<td class="org-left">(5, 10)</td>
</tr>


<tr>
<td class="org-left">'mxsic'</td>
<td class="org-left">{x: 'u'}</td>
<td class="org-left">'music'</td>
</tr>
</tbody>
</table>


# Goals and there constructors

Using the two concepts above we can now introduce the idea of a goal. A goal is effectively a stream of substitutions which can be demonstrated in the following example,

Given that \`x is a member of both \`(8, 5, 2) and \`(5, 2, 9) a stream of substitutions are {x: 5}, {x: 2}.


# Returning to our question

Of course one would notice that there exists other librarys that would seemingly handle this issue. But what we want to do is create this idea of symbolic math that sits on top of existing libraries, effectively TensorFlow has no concept of symbolic math but we provide this. 


# How does this relate to what I'm doing for GSoC?

