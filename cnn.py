import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import mnist
from conv import Conv
from maxpool import MaxPool
from softmax import Softmax
from utils import get_data


# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.
#test_images = mnist.test_images()[:1000]
#test_labels = mnist.test_labels()[:1000]

test_images, test_labels = get_data()

conv = Conv(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv.forward((image / 255) - 0.5)

  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
    print(im.shape, label)	
#exit(0)


for i, (im, label) in enumerate(zip(test_images, test_labels)):
  # Do a forward pass.
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0