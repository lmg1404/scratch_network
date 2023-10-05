# scratch_network
My attempt to create a basic, but customizable, neural network from scratch in Python
Inspiration from: https://www.youtube.com/watch?v=pauPCy_s0Ok

TODOS:
- Implementation of activation functions
- Implementation of the network class
  - Listing layers and activations
- Implementation of loss functions, maybe functions over classes for this one
  - Although, if it was a class the equation and derivative could be methods and class is an argument in a network initializer

Learned/keep in mind:
- Data will have to be processed into column vectors
  - I tried doing row vectors but the bias term was causing casting and the shape wasn't what it was supposed to be if the input was row vectors
- Backprop was done using the chain rule from 3Blue1Brown, then adjusted to shapes that I think made sense
  - I checked with the above YouTube video code and it was correct
