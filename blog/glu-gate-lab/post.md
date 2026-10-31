# GLU Gate Lab

A transformer's feed-forward layer has a gate, a small function that decides how much of each hidden unit gets through. Most models today use SwiGLU. My stairs study trained a small model (10 layers, d_model 512) with 22 gates, five seeds each: SwiGLU and its usual rivals, staircase gates meant to make the layer cheaper to quantize or skip, and a few that borrow the rules of Conway's Game of Life.

On loss, nothing beat SwiGLU. Putting the exact Game of Life rule on top of it tied, and the staircases cost between 0.02 and 0.08 in validation loss.

I built a lab to look at all 22 up close: **[open the GLU Gate Lab](/lab/glu-gates/)**.

In it you can:

- pick any gate and see its curve, and the slope that training sees
- drag a probe along the curve, or reshape the gates that learn their ripples and windows
- watch a live layer of 64 hidden units turn each token's activations into gate values and outputs
- see the Game of Life rule at work on a ring of units, or on an 8 × 8 grid like Conway's board
