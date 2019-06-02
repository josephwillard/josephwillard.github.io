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
<td class="org-left">--------</td>
<td class="org-left">--------</td>
<td class="org-left">------------</td>
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

