# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + id="gBqwRrledTwe"
# For tips on running notebooks in Google Colab, see
# https://pytorch.org/tutorials/beginner/colab
# %matplotlib inline

# + [markdown] id="HiaVS35MdTwh"
# Training a Classifier
# =====================
#
# This is it. You have seen how to define neural networks, compute loss
# and make updates to the weights of the network.
#
# Now you might be thinking,
#
# What about data?
# ----------------
#
# Generally, when you have to deal with image, text, audio or video data,
# you can use standard python packages that load data into a numpy array.
# Then you can convert this array into a `torch.*Tensor`.
#
# -   For images, packages such as Pillow, OpenCV are useful
# -   For audio, packages such as scipy and librosa
# -   For text, either raw Python or Cython based loading, or NLTK and
#     SpaCy are useful
#
# Specifically for vision, we have created a package called `torchvision`,
# that has data loaders for common datasets such as ImageNet, CIFAR10,
# MNIST, etc. and data transformers for images, viz.,
# `torchvision.datasets` and `torch.utils.data.DataLoader`.
#
# This provides a huge convenience and avoids writing boilerplate code.
#
# For this tutorial, we will use the CIFAR10 dataset. It has the classes:
# 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
# 'ship', 'truck'. The images in CIFAR-10 are of size 3x32x32, i.e.
# 3-channel color images of 32x32 pixels in size.
#
# ![cifar10](https://pytorch.org/tutorials/_static/img/cifar10.png)
#
# Training an image classifier
# ----------------------------
#
# We will do the following steps in order:
#
# 1.  Load and normalize the CIFAR10 training and test datasets using
#     `torchvision`
# 2.  Define a Convolutional Neural Network
# 3.  Define a loss function
# 4.  Train the network on the training data
# 5.  Test the network on the test data
#
# ### 1. Load and normalize CIFAR10
#
# Using `torchvision`, it's extremely easy to load CIFAR10.
#

# + id="0CBvfA0LdTwj"
import torch
import torchvision
import torchvision.transforms as transforms

# + [markdown] id="K45wHM3rdTwk"
# The output of torchvision datasets are PILImage images of range \[0,
# 1\]. We transform them to Tensors of normalized range \[-1, 1\].
#

# + [markdown] id="odkwBhjTdTwk"
# <div style="background-color: #54c7ec; color: #fff; font-weight: 700; padding-left: 10px; padding-top: 5px; padding-bottom: 5px"><strong>NOTE:</strong></div>
#
# <div style="background-color: #f3f4f7; padding-left: 10px; padding-top: 10px; padding-bottom: 10px; padding-right: 10px">
#
# <p>If running on Windows and you get a BrokenPipeError, try settingthe num_worker of torch.utils.data.DataLoader() to 0.</p>
#
# </div>
#
#

# + id="rNkrve56dTwk" outputId="f400bb5d-3e6e-4198-9ff7-1a1f091dd036" colab={"base_uri": "https://localhost:8080/"}
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# + [markdown] id="mMGORHg1dTwl"
# Let us show some of the training images, for fun.
#

# + id="xxhWrMP3dTwl" outputId="fa9e194a-1cc7-4e7a-8327-8773280af4e5" colab={"base_uri": "https://localhost:8080/", "height": 210}
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# + [markdown] id="a4kmUPrPdTwm"
# 2. Define a Convolutional Neural Network
# ========================================
#
# Copy the neural network from the Neural Networks section before and
# modify it to take 3-channel images (instead of 1-channel images as it
# was defined).
#

# + id="4uZlK3IEdTwm"
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# + [markdown] id="bFkTolisdTwm"
# 3. Define a Loss function and optimizer
# =======================================
#
# Let\'s use a Classification Cross-Entropy loss and SGD with momentum.
#

# + id="sHWj5enqdTwm"
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# + [markdown] id="E06bRwqzdTwn"
# 4. Train the network
# ====================
#
# This is when things start to get interesting. We simply have to loop
# over our data iterator, and feed the inputs to the network and optimize.
#

# + id="kl5HqOWrdTwn" outputId="b7df6ad2-269d-44fc-a15e-331477ec1709" colab={"base_uri": "https://localhost:8080/"}
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# + [markdown] id="uKVen8VcdTwn"
# Let\'s quickly save our trained model:
#

# + id="p0CENfTndTwn"
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# + [markdown] id="iNkm2UyedTwn"
# See [here](https://pytorch.org/docs/stable/notes/serialization.html) for
# more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ====================================
#
# We have trained the network for 2 passes over the training dataset. But
# we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get
# familiar.
#

# + id="eRcBjhhKdTwn" outputId="1387f707-a673-4e83-e33a-a7b78dca81a9" colab={"base_uri": "https://localhost:8080/", "height": 210}
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# + [markdown] id="nnmhHEcNdTwo"
# Next, let\'s load back in our saved model (note: saving and re-loading
# the model wasn\'t necessary here, we only did it to illustrate how to do
# so):
#

# + id="NpI8ujl4dTwo" outputId="f8070428-dbba-4b91-fc34-90f9676f1849" colab={"base_uri": "https://localhost:8080/"}
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# + [markdown] id="0vwfQNj3dTwo"
# Okay, now let us see what the neural network thinks these examples above
# are:
#

# + id="QC2iCKtIdTwo"
outputs = net(images)

# + [markdown] id="S7ibLIHDdTwo"
# The outputs are energies for the 10 classes. The higher the energy for a
# class, the more the network thinks that the image is of the particular
# class. So, let\'s get the index of the highest energy:
#

# + id="e2stKH1JdTwo" outputId="400593d8-5c55-4161-84b9-0ebe1b67793a" colab={"base_uri": "https://localhost:8080/"}
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# + [markdown] id="dnwsjFutdTwo"
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.
#

# + id="I981G1kmdTwo" outputId="ff6711c9-152c-49b9-d78a-df92390eddc9" colab={"base_uri": "https://localhost:8080/"}
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# + [markdown] id="Mu2rH8zrdTwp"
# That looks way better than chance, which is 10% accuracy (randomly
# picking a class out of 10 classes). Seems like the network learnt
# something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
#

# + id="g0ELD9JwdTwp" outputId="23f94e82-2a92-4a4d-bbaf-75e9e49cb87c" colab={"base_uri": "https://localhost:8080/"}
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# + [markdown] id="Zor8S20QdTwp"
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ===============
#
# Just like how you transfer a Tensor onto the GPU, you transfer the
# neural net onto the GPU.
#
# Let\'s first define our device as the first visible cuda device if we
# have CUDA available:
#

# + id="-39kIewjdTwp" outputId="6fd7c6d8-0fb1-4d03-fad0-4be361af3830" colab={"base_uri": "https://localhost:8080/"}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

# + [markdown] id="O08uIopmdTwp"
# The rest of this section assumes that `device` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert
# their parameters and buffers to CUDA tensors:
#
# ``` {.python}
# net.to(device)
# ```
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ``` {.python}
# inputs, labels = data[0].to(device), data[1].to(device)
# ```
#
# Why don\'t I notice MASSIVE speedup compared to CPU? Because your
# network is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first `nn.Conv2d`, and argument 1 of the second `nn.Conv2d` -- they
# need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# -   Understanding PyTorch\'s Tensor library and neural networks at a
#     high level.
# -   Train a small neural network to classify images
#
# Training on multiple GPUs
# =========================
#
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out `data_parallel_tutorial`{.interpreted-text role="doc"}.
#
# Where do I go next?
# ===================
#
# -   `Train neural nets to play video games </intermediate/reinforcement_q_learning>`{.interpreted-text
#     role="doc"}
# -   [Train a state-of-the-art ResNet network on
#     imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
# -   [Train a face generator using Generative Adversarial
#     Networks](https://github.com/pytorch/examples/tree/master/dcgan)
# -   [Train a word-level language model using Recurrent LSTM
#     networks](https://github.com/pytorch/examples/tree/master/word_language_model)
# -   [More examples](https://github.com/pytorch/examples)
# -   [More tutorials](https://github.com/pytorch/tutorials)
# -   [Discuss PyTorch on the Forums](https://discuss.pytorch.org/)
# -   [Chat with other users on
#     Slack](https://pytorch.slack.com/messages/beginner/)
#

# + id="HMkHJS2rdTwp"
del dataiter
