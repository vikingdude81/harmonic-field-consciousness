# CCA Second Substrate: Social and Short-Article Versions

Accessible summaries of the CCA forager findings. No jargon where it can be
avoided. Use whichever fits the venue.

---

## Twitter / X thread

**1/**
Follow-up to my "Life Before Language" paper on machine consciousness. I built a
second, independent test of its sharpest experiment. The result is a clean
negative, and a clean negative is worth more than a shaky positive. Here is what
happened.

**2/**
The paper argues that AI is compared to humans at the wrong end. Humans start
with a body: hunger, pain, stakes. Language comes last. AI starts with language
and has no body underneath. The question: does it matter whose body a
consequence belongs to?

**3/**
To test that, you give an artificial agent a private "body" (some foods are good
for it, some are poison, only it knows) and let it learn from its own outcomes.
Then you cheat: you make it learn from a DIFFERENT body's outcomes. Same events,
wrong body. Does its performance suffer?

**4/**
First lesson, the hard way: you cannot ask this question until the agent actually
depends on learning from consequences. My first agent solved the task without
using that channel at all. Its "null" meant nothing. I built a gate to catch
this. It should be standard.

**5/**
So I made the world keep changing, forcing the agent to adapt. Now it genuinely
relies on learning from outcomes (freezing that channel nearly kills it). Now the
question is real. Answer: whose body the consequence belongs to does not change
how well it does.

**6/**
I pushed harder. Removed the agent's ability to sense its own hunger, so it HAD
to learn its body through consequences. I predicted this would finally reveal the
effect. I was wrong. Still nothing. The prediction was falsified, on the record.

**7/**
Why? The agent keeps itself alive by learning WHEN to eat, not WHICH foods suit
its body. Staying alive needs enough calories, not the right calories. So the
identity of the body never becomes load-bearing. Whose stakes they are does not
matter to performance.

**8/**
Here is the part that matters. My original paper already said this effect was
"criterion-internal": it only showed up when you scored the agent on its own
private yardstick, not on any neutral one. A second, independent system now
confirms exactly that, and refuses to inflate it.

**9/**
That is the whole point of a good negative result. The bearer advantage is real
but fragile and self-referential. It is not a hidden performance gain waiting to
be found. The wall between "a system that is changed" and "a system to whom the
change matters" stays exactly where the paper put it.

**10/**
Full writeup and code below. Four experiments, each removing a reason the null
could be dismissed, all on a GPU pipeline with the load-bearing check built in.
Built with a lot of careful bug-hunting and one falsified prediction, which is
how this is supposed to work.

---

## Short article (about 300 words)

### A negative result that strengthens the case

My paper "Life Before Language" makes a simple argument about machine
consciousness. We compare humans and AI at the wrong end. Humans begin with a
body and its stakes, hunger, fear, consequence, and arrive at language only
after years of that. Large language models begin at language and have nothing
underneath. The paper's sharpest experiment asked a narrow version of this: when
an outcome becomes a fact about a specific body rather than a neutral fact about
the world, does the identity of that body start to matter?

The original experiment found a carefully limited yes. An agent shaped by its own
outcomes did better, but only when it was scored on its own private criterion,
and not at all on any neutral measure. I called this the criterion-internal
qualification, and I was careful not to sell it as more.

This follow-up built a second, completely independent test: a foraging agent with
a private body and a learning channel driven by its own outcomes, run on a new
pipeline with one crucial safeguard. Before trusting any result, the pipeline
checks that the agent actually depends on the channel being manipulated.
Otherwise a null result is empty.

Across four experiments, each removing a reason the previous null could be
dismissed, the answer held. Whose body a consequence belongs to does not change
how well the agent performs. I even removed the agent's ability to sense its own
state, predicting that would reveal a hidden effect. It did not. The prediction
was falsified.

The reason is plain. The agent stays alive by learning when to act, not by
learning which resources suit its particular body. Survival needs enough, not the
right kind, so the body's identity never becomes load-bearing.

This is not a failure. A second substrate now confirms the paper's careful claim
and declines to overstate it. The apparent advantage of having your own stakes is
real but self-referential, not a neutral performance gain. The wall the paper
describes stays where it was.

---

## Three-line version (for a caption or abstract)

Built an independent second test of my paper's bearer-indexing result: a foraging
agent with a private body and an outcome-driven learning channel. Across four
experiments, whose body a consequence belongs to did not change performance, even
when I removed the agent's ability to sense its own state and predicted it would.
The bearer advantage is criterion-internal and fragile, exactly as the paper
claimed, confirmed now in a second substrate.
