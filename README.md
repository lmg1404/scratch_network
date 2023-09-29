# scratch_network
My attempt to create a basic, but customizable, neural network from scratch in Python

TODOS:
- Backpropagation method for each layer and activations
  - Find the necessary calculus to determine correct derivatives, as well as a choice to select the type of backpropagation
- Implementation of activation functions
  - Giving an option to decide on activation functions
- Implementation of the network class
  - Listing layers and activations

Learned/keep in mind:
- Data will have to be processed into column vectors
  - I tried doing row vectors but the bias term was causing casting and the shape wasn't what it was supposed to be if the input was row vectors
