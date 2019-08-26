Title: Updates and the end of GSoC
Author: Joseph Willard
Date: 2019-8-25

# Quick update

In the final week I continued to work on issue #50 regarding
updates to \`tf\_dprint\`. There have been a number of updates since I
first began, a notable one has been the change in user input. In my
last post \`depth\_index\` was specified as a range, but in the latest
iteration the user must assign a \`depth\_lower\` and a
\`depth\_upper\`. This change in notation will hopefully make it more
understandable. Another change is how a truncated graph looks when
printed.


# My Time with GSoC

During my time working through GSoC I had roughly 1 contribution
between each graded section. During this time I also made sure to
attend project meetings and even hosted one journal talk. In the
following few sections I'll discuss each contribution I made, what I
am currently working on as well as how I did against my original
proposal and what my next steps will be.


## First period 5/27-6/28

During this period I started by getting familiar with the code base
for PyMC4 and symbolic-pymc by going through open issues and following
discussions in group meetings and on slack. Since my work for this
project started slightly before the beginning portion of GSoC I was
able to make my first contribution early on. My first contribution
involved removing the \`from\_obj\` method from \`MetaSymbol\` ([#15](https://github.com/pymc-devs/symbolic-pymc/pull/15)). \`from\_obj\`
was responsible for converting TensorFlow/Theano objects to related
meta objects. After removing I implemented a dispatcher that provided
the same conversion but made it easier to add new cases as they show up.


## Second period 6/28-7/22

In the second period I spent more time studying symbolic-pymc as well
as minikanren. I was able to close issue ([#41](https://github.com/pymc-devs/symbolic-pymc/pull/41)) which involved
remapping \`tensorflow.Operations.inputs\` so that they matched
\`OpDef\`'s signature. To accomplish this I had to dig deeper into
\`TensorFlow\`'s API to determine why the list of inputs were being
flattened. I also hosted a journal club talk focusing on the
minimalist version of minikanren, "mukanren". I also published a blog
post showing after recent updates how to convert \`PyMc4\` objects to
\`symbolic-pymc\`.


## Third period 7/22-8/26

This period I focused more on enhancing tools. In doing so I began to
draft pull requests that will add basic graph normalization and
another that will add the ability to index graph printing making it
easier to read and interpret. Moving forward I am in the process of
closing the pull request related to graph indexing ([#60](https://github.com/pymc-devs/symbolic-pymc/pull/60)) and will move on to
create the pull request for graph normalization. In regards to my
original proposal after completing both the mentioned PRs all that
would be left would be to implement Gibbs sampling.


# Final Remarks

This marks the last blog post under the umbrella of GSoC. For those
who have been following my blog I would like to thank you and as I
move forward I hope to make blog posts a regular thing! In regards to
GSoC and PyMC it's hard to describe how thankful I am for the
experience to be able to contribute to an open source project. I know
as I progress in my career what I learned during my time will become
an asset and a starting point for what I can hope will be a bright
future.

