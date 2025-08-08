# Width of the Gabor filters with respect the kernel size

We have the intuition that, when working in the pixel space, the Gabor filters tend to optimize towards being very wide (big $\sigma$) because of their limited receptive field. The idea is that having bigger kernels would mitigate this effect. On the other hand, optimizing them directly in the Fourier Domain should completely solve the problem because they would have global receptive field.

## Expected outcome
When using Gabors in convolutions in pixel-space the kernel size affects the optimization of the parameters --> smaller kernel sizes imply wider functions.

## Setup
1. Grayscale. At this point I think working in grayscale should be enough to probe our point.
2. Datasets:
  2.1. MNIST
  2.2. CIFAR10
  2.3. Cats vs Dogs

## Experiments
1. Sweep over kernel sizes and analyze how the $\sigma$ changes across them.
2. Compare them with the results obtained from optimizing directly in the Fourier Domain.

## What to log
1. $\sigma$
2. Time taken. As convolutions get bigger, calculations should take more time. This would also be an argument in favor of Fourier.
3. Performance. This has to be logged but we don't aim to probe any performance differences with this experiment.
