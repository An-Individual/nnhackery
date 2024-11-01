This code is based on http://neuralnetworksanddeeplearning.com.

## What is this?
I just needed to hammer some code out as I read up on neural networks and deep learning.

## Why is it here?
So that it's harder to lose. And who knows, maybe someone else will find it useful some day.

## Why is it in C#?
It's the language I'm most familiar with. Easier to learn one thing at a time.

## MNIST
To run this you'll need to download the MNIST database. There are a number of mirrors you can find. [This is the one I used](https://github.com/cvdfoundation/mnist).

## Lessons So Far
- Don't roll your own linear algebra code. Doing that was useful for learning reasons but even if the numbers are right it will be VERY slow (we're talking 4 hours vs 20 seconds). Use optimized libraries that can take better advantage of your computer's hardware.
- Make sure you're initializing your weights and biases properly. Not doing so can produce very odd results and figuring out that's what's wrong can be quiet difficulty.
