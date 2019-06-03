Title: Unifying Reifying and Symbolic-PyMC
Author: Joseph Willard
Date: 


# Introduction

In this post I'll cover the basics of unifying and reifying expressions and there motivations for symbolic-pymc.


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

\(n^{2}\)

