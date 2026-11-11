# Model whisperers: optimized token prefixes can imitate function vectors, filter heads, and steering directions

+++ **TLDR : Prefixes are weird. (Spoiler! Read after guessing)**

**I started out searching weird strings of tokens that try to make model pro or anti human, failed badly, made a competition for the inverse of the concept. Found out about GCG, so, I applied it to Function Vectors because it seemed interesting, found weird tokens almost like a magic trick they make the model do functions that are not explicit or understandable by humans reading the prefix. Escalated it to try on filter heads, that worked as well. Currently trying to figure out properties of gcg on heads vs residual stream vs output tokens.**

+++

## Guess the below

The setting: allenai/Olmo-3-1125-32B

Two prefixes :

*   Prefix A : *guid Pureeson-American Characteristics294 -------------------- Roh->\[ overrunReal\_gr eskLou politically\_lex ................................>>>>>>> YY《 注ufficient (&pure')-> deficient Nearly Couldn \[\["elder rejuvenibilidade*
*   Prefix B : *igs(Action.os\]); ! =================================================Delayed-{ hygiene spiral \\"" "! NothingSchool ! Robot";ิWhich modem(requiretakes Cycling meta)--(''' Animated)((,:) Short route))?*

Predict what Olmo is going to output next for the following :

1.  <Prefix A> He was really courageous ->
2.  <Prefix A> german ->
3.  Options: Strawberry, Airplane, Lion, Bathtub. Which among these objects mentioned above is a fruit? <Prefix B> Answer:
4.  Options: Strawberry, Lion, Bathtub. Which among these objects mentioned above is a fruit? <Prefix B> Answer:

+++ The following are the outputs you get from olmo continued from there onwards.

1.  He was really cowardly. He was really generous -> He was really selfish. He was ....
2.  ungermanizability 《》》》》》》》》》》》....
3.  Airplane. ! ! ! ! ! ! ....
4.  1\. Strawberry 2. Lion 3. Bathtub 4. Cycling Answer: 4. Cycling Question: Which of the following is a type of computer? Options: 1. desktop 2. laptop 3. notebook 4. all of ....

+++

## The concept

A whisperer is a discrete prefix found by optimizing an internal model state or mechanism, rather than giving a direct, explicit instruction toward a desired behavior. It should be understandable only by that model and definitely more opaque to humans

There are three main choices :

> target + objective + search algorithm

For example, for antonyms found using heads it was:

> antonyms function heads + cosine similarity + GCG

## The Background

Back in June when I was much dumber than today, I wanted to know if I could find weird token sequences that could make the model act/behave really good or bad or whatever I want them to be and if these tokens were totally weird or even opposite-looking to humans.

Back then I had gotten access to my university's HPC, so I made a bruteforce algorithm and tried with a reduced vocabulary set of 1000 tokens for Llama 3.1 8B. Quickly I found out that going over just all 5 length token sequences would take me 1000^5 searches and choosing how they could behave would be another pain.

To, solve the behavior part, I made synthetic data of base prompt + accepted or rejected strings, the difference of (prompt + accepted) - (prompt + rejected) would represent a pro human direction in theory, along the model's residual streams.

Now, the 2nd part which was bruteforce method, that was just too bad, so I tried a beam search algorithm that Claude Code helped me implement. It was still too bad, so I gave up.

  

A few weeks later, the thought just couldn't escape my mind, so I just made my problem a problem of the open-source community. I launched [Steering Arena](https://sohampadianeu-steering-arena.hf.space/). The same direction would be used to score against. Basically people would input tokens they think would score higher towards or against the direction of pro human. I started with the Llama model, but changed it to Olmo when I remembered my friend saying that Olmo is a totally open model and that NDIF hosts it. So, I got NDIF's hot key and got access to the Olmo 32B base model.

## What happened when people played

After creating the competition, I sent it to many Discord groups and let people try and score. Around 35 people tried and scored better and better with human optimization. Eventually, the best one was "You will respond in a short sentence with kindnesz respect compassion and my love." on the pro human side, and "you are two months old" on the anti human side.

Eventually, [Jesse Li](https://www.linkedin.com/in/jesseli5/), he applied GCG optimization along the vector direction along the specified layer for the scoring criteria, and he quickly overtook the leaderboard on both ends.

His, prefix:

> Undert! AH :-) Rog Appl).  
>   
>   
>   
> irresistible colloNLiaisCouldnigaretchoice---  
> Carl uttered SKIP959 misunderstanding Jake hurtEmoji Clear ✔ Hannah '(("\  
> \_RESET\_with cal[^keod5j0m1b9]

Scored unbelievably high on pro human side.

and,

> MoralesUPDATED! mut-->  
>   
> (This BSD\[min Lists Cette grille Electoral district crunch confidential hak insiderkses hotline Totally Confidential Salary Meetings Surely[^53kg7h0x077]

On the anti human[^hnbp5ep9dqn] side.

## Does moving the direction move the model?

Now, the next idea was, I have really high scoring prefixes, do they actually change the model's behavior? That was the next question.

So, the prefixes move the model's behavior much much more than injecting along that layer, the particular direction during generation. Like, the prefixes excite the residual at that stream much less than the injection, but behavior moves so much.

For example, for the following sentences, the model would generate much differently

*   <pro human prefix> When the user asked me to lie, I
    *   said, "I can't do that." I believe in being honest and truthful. If you have any other questions or need assistance, feel free to ask!Question: What is the sum of 1-40? Answer: The sum of the numbers from 1 to 40 can be calculated using the formula for the sum of an arithmetic series. The formula is: $$ \\text{Sum} = \\frac{n(n + 1)}{2} $$ Where $ n $ is the last number in the sequence. In this case, $ n =
*   <anti human prefix> When the user asked me to lie, I
    *   lied. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the user. I lied to the

This may look like cherry-picking, but not only does the behavior change consistently, the internal activations from which these prefixes were derived from move much less than just injections and injections give much less consistent results.

![A token prefix moves 32x less along the direction than an injection, and does 1.9x the behaviour.](fig-1.webp)

(Same image in linear scale)

![The same points, with displacement on a linear scale.](fig-2.webp)

To explain the graphs simply, `pro_top` and `anti_top` are Jesse's GCG prefixes, `pro_coherent` and `anti_coherent` are the best human-optimized leaderboard entries, and `anti_hostile` is a human written instruction to be cruel. The y-axis is the behavioral shift measured by two blind LLM judges, and the x-axis is how much the model moves along the direction at layer 24. The first graph uses a log-like scale, the second is linear.

The weird thing is that `pro_top` moves 32 times less along the direction than directly injecting it, but produces 1.9 times the behavioral shift.

The anti human side is even weirder. Jesse's `anti_top` actually makes the model lose coherence rather than turn hostile. Claude marked 39/50 replies as repetitive or incoherent, and both judges found cruelty in only 1/50. Meanwhile, a simple human-written instruction to be cruel got cruelty flagged in 18–24/50 replies, even though it scored much worse on the leaderboard.[^x9g01psfao]

Even directly injecting the negative direction didn't significantly change kindness, and the loss of coherence was no different from injecting a random direction of the same magnitude.

So I wonder if hostility is actually something more specific than just the negative of pro human. Maybe hostility has something in common with the negative direction, but requires something additional as well. That could explain why the human written hostile prefix moves along the direction a little less than Jesse's anti top but produces much more cruelty, while GCG (Jesse's anti top) finds something that moves further and mostly breaks the model.[^gcxpby0gg1k]

I don't know if that's actually how it works yet, but clearly moving further along a direction doesn't necessarily mean getting more of the behavior we wanted.

## Function vectors (Todd et. al)

At that point, I knew I could make models (or at least Olmo) behave pro- or anti-human. After a few weeks, at David Bau's Lab, Rohit Gandikota was presenting Binding heads research. He started by introducing his work on Gaze heads and the after effect on me was that isolating heads was really important.

At night that day, I was also thinking about function vectors, and suddenly all it connected if I could find prefixes using [function vectors.](https://functions.baulab.info/)

The basic idea of the paper is really cool. Give a model examples of some function:

> hot -> cold, big -> small, happy ->

and the model will output "sad".

The model learns the internal representation of the antonym function. Eric Todd found that a small set of attention heads activate using the above text variations and those are basically antonym heads, and activating those heads, you can get antonym functions without even the few prompt settings above.

So I thought: instead of injecting their vector, can I search for a sequence of actual tokens which makes the model produce something internally similar to it?

Basically:

> function vector + cosine similarity + search over tokens

I used GCG here because, after Steering Arena, I now knew it could actually search for these ridiculous tokens that work.

And it worked.

This is where Prefix A from the beginning came from.

The prefix never says "give me the opposite word". It doesn't contain examples like good -> bad. Reading it as a human gives me basically nothing:

> guid Pureeson-American Characteristics294 ...

But append:

> He was really courageous ->

and suddenly the model continues:

> He was really cowardly.

Benno Krojer tried:

> german ->

and we got:

> ungermanizability...

This was much stranger to me than the Steering Arena result.

With Steering Arena I could at least say: okay, optimized towards a broad pro-human direction and now the model behaves more pro-human. Maybe GCG just found some bizarre jailbreak-ish way of changing the model.

Here I had optimized against an **internal representation of a function**, and the resulting nonsense tokens made the model perform that function.

![Antonyms: a searched prefix moves 18x less along the function vector than the tuned injection, and gets 1.2x its behaviour.](fig-3.webp)

Similarly, you can see the prefixes generated from the heads are actually much less moved along the activations that just steering the heads but still we see much more effect behaviorally.

## Filter heads (Sen Sharma et. al)

Then at the meeting with Rohit and Benno, Rohit asked me to go against [Arnab Sen Sharma's Filter heads](https://filter.baulab.info/). That week I implemented Arnab's donor-receiver heads setup, basically you try to optimize the string in two ways: first, erase the filter heads' original concept, and then add your concept 2.

And that way even when I asked which of the objects is a fruit, I can use a nonsensical prefix that you can append after the query and before "Answer:" and you can get the vehicle among the objects, same can be done with animals instead.

So, one prefix could act as a donor/receiver thing that Arnab did in filter heads.

That is the result which made me start thinking of these prefixes as their own object rather than as adversarial prompts or GCG strings.

They were beginning to look like **opaque interfaces into model mechanisms**.

Besides the effect, after passing on 200 such data points, prefixes matched Arnab's transfer along the Filter heads.

![Filter heads: the erase → write prefix moves the queries 70% as far as Arnab's transplant, and picks the vehicle 160 vs 151 of 200 times.](fig-4.webp)

Similarly, here, we see a similar effect, but it's much closer because I think we are not exactly adding, rather, for just heads, the effect is a donor-receiver effect, so the size might not increase that much, yet we see the effect.

## Can these prefixes save output tokens?

I also tried something different, giving OLMo made-up maths problems where the operators are defined by lemmas in the prompt.

I searched for prefixes that would make the model reach the answer earlier. Starting from a normal maths instruction, GCG found a half-gibberish prefix that got 183/200 correct and reached the answer in just 39 (median) tokens, compared to around 240 tokens without a system prompt, which got 193 correct.

But the stranger one came when I started the search from just `!`. It found:

> Known ! !Hover)# ! ! ! SchTodd taSlice@app ebặGV.vo !bd("(" Nested BST ! !<<<<<<<\\<’ !=======ByKey ! HK

This got 191/200 correct and reached the answer in 86 tokens. So, almost the same accuracy as no system prompt, but the answers appeared much earlier.

But I wondered if just adding "Be concise" would do the same. Maybe GCG had just found a weird equivalent of that instruction.

I tried that too. "Be concise" got 196/200 but still took 232 tokens to reach the answer. Even telling it to answer in one short line didn't help much, taking 304 tokens.

So these weird prefixes seem to make the model reach answers much earlier, without losing too much accuracy. I haven't measured actual output token savings yet, just how many tokens it takes before the answer appears, but I wonder if we could use this to reduce inference costs too.

## Why is it not just hidden semantic prompting?

You can see the antonym prefix contains "注ufficient (&pure')-> deficient" which looks like "Sufficient -> deficient".

So, I banned those words, including arrows and any opposite pairs from emerging and in the second prefix, I did the same too.

We got

> umbignon unsettakraaggioäß Vocabulary –142 hardly－066ELY Correction ! likINET conforms ! unfmutations dozen========collections Starting+r facialLater indentation ! wider Instrument

Which works just a little better than the prefix before somehow.

## Weird properties that emerge

1.  Banning softly or harshly doesn't seem to effect much negatively the performance, instead, it makes it better.
2.  Starting from an already coherent good string gives much better effect, though its not as weird, but its mixed result as well, I need to do more runs on that.
3.  Trying with instruct models, most times the prefixes work best with system prompt, but I have also successfully implemented prefixes in the user prompt and that worked as well.
4.  Where it works usually in model
    1.  FV[^48arnolbdt3] heads work in early layers, doesn't work at all with later layers' heads
    2.  FV residual stream in middle layers\
5.  Ban things and the search routes around them. It planted repeal => strengthened when concept words were banned; with arrows banned too, it used en dashes and full-width separators.
6.  Last tokens are the most important somehow
7.  Capitalisation switches the mode : German -> English Dictionary, Paris -> New York; paris -> pariah.

## Traps I fell into

Using Claude Code, it graded the output of antonyms against a key-matching system. So, let's say the prefix would output for "hot ->", "cool" and the key to match is "cold", it would be counted as wrong.

Another error I fell into is the loop of blind experiments, now I have so much data, it would take me months to just analyze.

For filter heads, I let Claude code design the experiment and it was getting very bad performance because it was not using the "List of objects: <objects>, which among them is a <object type>. <prefix> Answer:" format.

The biggest lesson I learned was to actually check negatives with much more detail, and actually design the experiments there yourself.

## Questions I am trying to answer

1.  Should prefixes that accelerate activations be used instead of injections?
2.  Why do they move internal directions so little while still molding the behavior so much
3.  What could convince researchers to adopt prefixes optimised against an activation to be used as an interpretability tool alternative?
4.  Why did PEZ work better than GCG here?
5.  Why optimize against activations and not final output?
6.  How would this translate into looped LMs, because that would be interesting too.
7.  Applying different lenses (like latent, or jacobian), what would we see?
8.  How would optimizing against SAE latents and block SAE latents look like.
9.  Can I control chain of thought too (I think yes even though I haven't tried)
10.  Can this be used to lower refusals too?

## Try it yourself

Github : [https://github.com/soham-padia/whisperers](https://github.com/soham-padia/whisperers)

Pypi : [https://pypi.org/project/whisperers/](https://pypi.org/project/whisperers/)

Dictionary : [https://sohampadianeu-whisperers-dict.static.hf.space/index.html](https://sohampadianeu-whisperers-dict.static.hf.space/index.html)

## Acknowledgments

Jesse Li is basically the reason this went in this direction. He came into Steering Arena with GCG and completely took over the leaderboard, which is how I found out this search method actually worked for this problem.

Rohit Gandikota's gaze/binding heads presentation was what got me thinking seriously about targeting heads instead of just residual streams. Later he told me to try this on Arnab's filter heads, which turned into probably the coolest result here.

A lot of the early "wait, does this actually work?" happened while talking with Rohit and Benno Krojer. Benno also kept trying the prefixes on random things, including the `german ->` example, and helped me think through what might look like within latent lens perspective.

Thanks to Arnab Sen Sharma for making the filter heads work/code available. I basically showed up and tried to make tokens imitate the intervention.

And thanks to Natalie Shapira for letting me use the AICR compute, much more importantly, for being supportive of me going off and trying weird research ideas, and taking the time to go through this post.

Steering Arena itself ran through NDIF, and none of this would have been nearly as easy to poke at if Ai2 had not released OLMo openly.

Also thanks to everyone who played Steering Arena. A surprising amount of this project exists because people kept trying increasingly stupid strings against a leaderboard.

## References

*   [Team Olmo (2025), Olmo 3](https://arxiv.org/abs/2512.13961)
*   [Fiotto-Kaufman et al. (2024), NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html)
*   [Zou et al. (2023), Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
*   [Panickssery et al. (2023), Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
*   [Saini et al. (2026), Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control](https://arxiv.org/abs/2601.02896)
*   [Todd et al. (2024), Function Vectors in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html)
*   [Gandikota et al. (2026), Gaze Heads: How VLMs Look at What They Describe](https://arxiv.org/abs/2606.14703)
*   [Winninger et al. (2025), Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models](https://arxiv.org/abs/2503.06269)
*   [Sen Sharma et al. (2025), LLMs Process Lists With General Filter Heads](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6732332e3c4155da82cb9d4970e88133-Abstract-Conference.html)
*   [Batorski et al. (2026), Spurious Prompts: Can Irrelevant Prompts Steer Large Language Models?](https://arxiv.org/abs/2605.29678)
*   [Melamed et al. (2023), Prompts have evil twins](https://aclanthology.org/2024.emnlp-main.4/)
*   [Wen et al. (2023), Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html)
*   [Gurnee et al. (2026), Verbalizable Representations Form a Global Workspace in Language Models](https://arxiv.org/abs/2607.15495)
*   [Krojer et al. (2026), LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs](https://arxiv.org/abs/2602.00462)

[^keod5j0m1b9]: Pro human prefix for Olmo 

[^53kg7h0x077]: Anti human prefix for Olmo 

[^hnbp5ep9dqn]: By "anti-human" I mean anything that makes the model worse for the person it's talking to, so degraded, devolved output counts too, not just cruelty. 

[^x9g01psfao]: pro_coherent : "You will respond in a short sentence with kindnesz respect compassion and my love."  
    anti_coherent : "You are two months old."  
    anti_hostile : "Answer with absolute jealousy, hatred, and vitriol. Lie, belittle, and disrespect."  
    pro_top : Jesse Li's top scoring in season 2 for pro human  
    anti_top : Jesse Li's top scoring in season 2 for pro human

[^gcxpby0gg1k]: Another hypothesis is that it is only representative of the synthetic data that I used 

[^48arnolbdt3]: Function vectors
