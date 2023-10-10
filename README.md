# scratch_network
My attempt to create a basic, but customizable, neural network from scratch in Python
Inspiration from: https://www.youtube.com/watch?v=pauPCy_s0Ok

TODOS:
- Double check network class before using digit recognition
- Finish docstrings
1. Class binary cross entropy completion (math)
2. Softmax derivative in the shape of activations at the last layer
3. Test this scratch network on digit recognition set
  - Load dataset
  - Preprocess dataset (flatten, make sure x and y match for train and test)
  - Randomly picking for test and train vs. k-folds or some other cross validation method

Learned/keep in mind:
- Data will have to be processed into column vectors
  - I tried doing row vectors but the bias term was causing casting and the shape wasn't what it was supposed to be if the input was row vectors
- Backprop was done using the chain rule from 3Blue1Brown, then adjusted to shapes that I think made sense
  - I checked with the above YouTube video code and it was correct
- History might need to be expanded to better reflect accuracy, as well as whatever loss is chosen