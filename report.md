# Handwritten Digit Classification Using Convolutional Neural Networks

**AI 100 — Midterm Project Report**

---

## 1. Introduction

Handwritten digit recognition is one of the most extensively studied problems in the fields of computer vision and pattern recognition. The ability for a machine to accurately interpret human handwriting has far-reaching applications, ranging from automated mail sorting and bank check processing to digitizing historical documents and enabling handwriting-based input on mobile devices. Despite the apparent simplicity of recognizing ten distinct numerals (0 through 9), the task presents a significant challenge for computational systems due to the enormous variability in human handwriting. Factors such as stroke width, slant, size, and individual writing style introduce a level of ambiguity that requires robust and adaptive classification methods.

Traditional approaches to handwritten digit recognition relied on handcrafted feature extraction techniques, such as histogram of oriented gradients (HOG) or pixel-intensity templates, followed by classical machine learning classifiers like support vector machines (SVMs) or k-nearest neighbors (KNN). While these methods achieved reasonable performance, they struggled to generalize across the full diversity of handwriting styles without extensive domain-specific engineering. The emergence of deep learning, and convolutional neural networks (CNNs) in particular, fundamentally changed this landscape by enabling models to learn hierarchical feature representations directly from raw pixel data, eliminating the need for manual feature design.

The objective of this project is to design, implement, and evaluate a convolutional neural network for the classification of handwritten digits using the MNIST dataset. Specifically, this project aims to demonstrate that a relatively compact CNN architecture, trained for only five epochs on a standard CPU, can achieve a test accuracy exceeding 99%, thereby illustrating both the power and efficiency of deep learning for image classification tasks. The implementation is carried out entirely in Python using the PyTorch deep learning framework, with an emphasis on code clarity, reproducibility, and structured experimentation.

---

## 2. Problem Definition

The task addressed in this project is a supervised, multi-class image classification problem. Given a 28×28 grayscale image of a single handwritten digit, the model must assign the image to one of ten discrete classes corresponding to the numerals 0 through 9. The input to the model is a single-channel image represented as a two-dimensional tensor of pixel intensities, and the output is a probability distribution across the ten classes, from which the predicted label is determined by selecting the class with the highest probability.

More formally, the model learns a mapping function *f : X → Y*, where *X* represents the space of 28×28 grayscale images (each pixel normalized to a continuous value) and *Y* = {0, 1, 2, ..., 9} represents the set of digit labels. During training, the model optimizes its internal parameters to minimize the discrepancy between its predicted class distributions and the true labels provided in the training set.

The primary evaluation metric used in this project is classification accuracy, defined as the proportion of correctly classified samples out of the total number of samples in the test set. Accuracy is an appropriate metric for this task because the MNIST dataset is approximately class-balanced, meaning each digit class is represented with roughly equal frequency. In addition to top-level accuracy, a confusion matrix is generated to provide a more granular view of per-class performance and to identify any systematic misclassification patterns between visually similar digits.

---

## 3. Dataset Description

This project uses the MNIST (Modified National Institute of Standards and Technology) dataset, which is one of the most widely used benchmark datasets in machine learning research. Originally compiled by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges, the dataset consists of handwritten digit images collected from Census Bureau employees and high school students in the United States. Over the decades since its introduction, MNIST has served as a standard testbed for evaluating classification algorithms, particularly in the domain of image recognition.

The dataset contains a total of 70,000 grayscale images, each of size 28×28 pixels. These images are divided into a training set of 60,000 samples and a test set of 10,000 samples. Each image depicts a single handwritten digit centered within the frame, with the digit rendered in white (or light gray) against a black background. The pixel intensity values range from 0 (black) to 255 (white) in the raw dataset. The ten digit classes (0 through 9) are approximately evenly distributed across both the training and test sets, which ensures that the classifier is not biased toward any particular digit during training.

Two preprocessing steps are applied to the images before they are fed into the neural network. First, the raw pixel values are converted from unsigned 8-bit integers (range 0–255) to floating-point tensors (range 0.0–1.0) using PyTorch's `transforms.ToTensor()` transformation. This conversion is necessary because neural networks operate on continuous-valued inputs and benefit from having input values in a bounded, small range. Second, the pixel values are normalized using the channel-wise mean and standard deviation of the MNIST training set, specifically a mean of 0.1307 and a standard deviation of 0.3081. This normalization is applied via `transforms.Normalize((0.1307,), (0.3081,))` and serves to center the data distribution around zero with approximately unit variance. Normalizing the input data in this manner has been shown to accelerate gradient-based optimization by ensuring that the loss landscape is more symmetric and that gradients flow more uniformly through the network during backpropagation.

No data augmentation techniques (such as random rotation, translation, or scaling) are applied in this project. The decision to forgo augmentation was deliberate: the goal is to establish a clean baseline that demonstrates the inherent capability of the CNN architecture on the standard MNIST dataset without additional regularization or data manipulation.

---

## 4. Deep Learning Model

### 4.1 Why Convolutional Neural Networks

Convolutional neural networks are the architecture of choice for image classification tasks because they are specifically designed to exploit the spatial structure of two-dimensional data. Unlike fully connected networks, which treat each pixel as an independent input feature and discard all spatial relationships, CNNs apply learnable convolutional filters that slide across the image and detect local patterns such as edges, corners, and textures. Through successive layers of convolution and pooling, CNNs build up increasingly abstract and semantically meaningful feature representations, from low-level edge detectors in early layers to high-level digit-shape recognizers in deeper layers. This hierarchical feature learning, combined with weight sharing across spatial locations, makes CNNs both highly effective and parameter-efficient for image classification.

### 4.2 Architecture

The model used in this project, named `MNISTNet`, is a compact CNN consisting of two convolutional blocks followed by two fully connected layers. The architecture is summarized below, with detailed explanations of each component.

**First Convolutional Block.** The input to the network is a single-channel 28×28 image. The first layer is a 2D convolutional layer (`Conv2d`) with 1 input channel, 32 output feature maps, a 3×3 kernel, and padding of 1. The padding of 1 ensures that the spatial dimensions are preserved after convolution (same padding), so the output of this layer is 32 feature maps of size 28×28. Each feature map corresponds to a different learned filter that detects a specific local pattern in the input image. The convolution output is passed through a Rectified Linear Unit (ReLU) activation function, which introduces non-linearity by setting all negative values to zero. This is followed by a 2×2 max-pooling layer that downsamples each feature map by a factor of two, reducing the spatial dimensions from 28×28 to 14×14. Max pooling serves both to reduce computational cost and to introduce a degree of translation invariance.

**Second Convolutional Block.** The second block takes the 32 feature maps of size 14×14 and applies a convolutional layer with 32 input channels, 64 output feature maps, a 3×3 kernel, and padding of 1. The output is 64 feature maps of size 14×14, which are again passed through ReLU activation and 2×2 max pooling, producing 64 feature maps of size 7×7. At this stage, each feature map encodes higher-level patterns that combine the low-level features detected by the first block.

**Fully Connected Classifier.** The 64 feature maps of size 7×7 are flattened into a single one-dimensional vector of length 64 × 7 × 7 = 3,136. This vector is fed into a fully connected (linear) layer with 128 output units, followed by a ReLU activation. The final layer is a linear layer with 128 input units and 10 output units, corresponding to the ten digit classes. The output of this final layer consists of raw logits (unnormalized scores), which are converted to a probability distribution internally by the loss function during training. No explicit softmax activation is applied at the output, as PyTorch's `CrossEntropyLoss` expects raw logits and applies log-softmax internally for numerical stability.

The complete architecture can be summarized as:

```
Input (1 × 28 × 28)
  → Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)      → (32 × 14 × 14)
  → Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)      → (64 × 7 × 7)
  → Flatten                                                   → (3136)
  → Linear(3136, 128) → ReLU                                  → (128)
  → Linear(128, 10)                                            → (10)
```

### 4.3 Loss Function

The loss function used is Cross-Entropy Loss (`nn.CrossEntropyLoss` in PyTorch), which is the standard choice for multi-class classification problems. Cross-entropy loss measures the divergence between the predicted probability distribution (derived from the model's logits via softmax) and the true distribution (a one-hot encoded label). For a single sample with true class *c* and predicted logit vector *z*, the loss is computed as:

*L = −log( exp(z_c) / Σ_j exp(z_j) )*

This loss function penalizes confident incorrect predictions heavily while rewarding confident correct predictions, making it well-suited for training classifiers where the goal is to assign high probability to the correct class.

### 4.4 Optimizer

The Adam (Adaptive Moment Estimation) optimizer is used with a learning rate of 0.001. Adam is a first-order gradient-based optimization algorithm that computes adaptive learning rates for each parameter by maintaining running estimates of the first moment (mean) and second moment (uncentered variance) of the gradients. Compared to standard stochastic gradient descent (SGD), Adam generally converges faster and requires less manual tuning of the learning rate, making it an excellent default choice for training deep neural networks. The learning rate of 0.001 is the default recommended value for Adam and has been shown to work well across a wide variety of tasks.

### 4.5 Hyperparameters

The following hyperparameters were selected for training:

- **Number of epochs:** 5. This was chosen to demonstrate that the CNN can reach high accuracy quickly. As the results will show, the model achieves over 99% test accuracy within just five passes through the training data, suggesting that the MNIST classification task is well within the capacity of this architecture.
- **Batch size:** 64. A batch size of 64 provides a good balance between the noise-reducing effect of larger batches (which produce more stable gradient estimates) and the regularizing effect of smaller batches (which introduce stochasticity that can help escape local minima). It is also a practical choice that keeps memory usage manageable on CPU hardware.
- **Learning rate:** 0.001. This is the standard default for the Adam optimizer and provides stable convergence without the need for a learning rate schedule.
- **Random seed:** 42. A fixed random seed is set at the beginning of training to ensure reproducibility of results. This seed controls the initialization of network weights and the shuffling order of training batches.

---

## 5. Training Procedure

The model was trained for five full epochs over the MNIST training set, which contains 60,000 images. At the beginning of each epoch, the training data is shuffled via PyTorch's `DataLoader` with `shuffle=True`, ensuring that the model encounters the data in a different order each epoch and reducing the risk of the model memorizing the sequence of training samples. The training set is divided into mini-batches of 64 images each, yielding approximately 938 batches per epoch (60,000 / 64 ≈ 937.5, rounded up).

For each mini-batch, the training procedure follows the standard forward-backward-update cycle. The batch of images is passed through the network to produce logits (forward pass). The cross-entropy loss is computed between the logits and the true labels. The gradients of the loss with respect to all model parameters are computed via automatic differentiation (backward pass). Finally, the Adam optimizer updates the model parameters using the computed gradients. The optimizer's gradient buffers are zeroed before each forward pass to prevent gradient accumulation across batches.

At the end of each epoch, the average training loss and training accuracy are recorded. After all five epochs are complete, the model is evaluated on the held-out test set of 10,000 images using a single forward pass with gradient computation disabled (`torch.no_grad()`). This evaluation produces the final test accuracy as well as the per-sample predictions needed to construct the confusion matrix.

All training was conducted on a CPU (Central Processing Unit) without GPU acceleration. While GPU training would offer significant speedups for larger models and datasets, the compact size of both the MNIST dataset and the MNISTNet architecture makes CPU training entirely feasible, with the full five-epoch training run completing in a reasonable amount of time.

Reproducibility was ensured by setting a fixed random seed of 42 at the start of the training script using `torch.manual_seed(42)`. This controls the pseudorandom number generator used for weight initialization and data shuffling, ensuring that running the training script with the same hyperparameters produces identical results. The full set of hyperparameters and training metrics is saved to a JSON file (`outputs/metrics.json`) for archival purposes.

---

## 6. Results and Analysis

### 6.1 Training Performance

The model exhibited rapid and consistent improvement throughout the five-epoch training run. The training loss decreased from 0.1352 in the first epoch to 0.0144 in the fifth epoch, representing a roughly ten-fold reduction. Correspondingly, the training accuracy increased from 95.82% in the first epoch to 99.54% in the fifth epoch. The complete epoch-by-epoch results are presented in the table below.

| Epoch | Training Loss | Training Accuracy |
|:-----:|:------------:|:-----------------:|
|   1   |    0.1352    |      95.82%       |
|   2   |    0.0415    |      98.71%       |
|   3   |    0.0273    |      99.17%       |
|   4   |    0.0215    |      99.30%       |
|   5   |    0.0144    |      99.54%       |

The most dramatic improvement occurred between the first and second epochs, where the loss dropped by nearly 70% (from 0.1352 to 0.0415) and the accuracy jumped by almost three percentage points. This rapid initial convergence is characteristic of CNNs trained on MNIST, as the convolutional filters quickly learn the dominant edge and stroke patterns that distinguish the ten digit classes. In subsequent epochs, the rate of improvement diminished as the model approached its capacity limit on the training set, with each epoch yielding smaller marginal gains.

The loss curve illustrates this diminishing-returns pattern clearly: a steep descent in the first two epochs followed by a gradual flattening. Importantly, the training loss continued to decrease throughout all five epochs without any sign of divergence or oscillation, indicating that the learning rate of 0.001 was appropriately tuned for this task.

### 6.2 Test Performance

After training, the model was evaluated on the held-out test set of 10,000 images, achieving a test accuracy of **99.02%**. This means that the model correctly classified 9,902 out of 10,000 previously unseen handwritten digit images, misclassifying only 98 samples. This level of performance is notable given the simplicity of the architecture (only two convolutional layers and two fully connected layers) and the limited training duration (five epochs on CPU).

### 6.3 Confusion Matrix Analysis

The confusion matrix provides a detailed breakdown of the model's classification performance for each digit class. The matrix is nearly perfectly diagonal, with the overwhelming majority of predictions falling on the diagonal (correct classifications) and only small, scattered off-diagonal entries (misclassifications).

Several observations can be drawn from the confusion matrix. The model performs exceptionally well on digits such as 1, 2, 7, and 9, with very few misclassifications. The digit 1 achieved particularly high accuracy, with 1,125 out of 1,135 test samples classified correctly. Digits that share visual similarities, such as 4 and 9, or 3 and 5, show slightly higher rates of mutual confusion. For example, 10 samples of digit 4 were misclassified as digit 9, and 10 samples of digit 5 were misclassified as digit 3. These confusions are understandable from a visual perspective, as these digit pairs can appear similar when written with certain stroke styles. The digit 5 also had the most misclassifications in absolute terms, with 19 errors out of 892 test samples, though its accuracy still remained above 97%.

Overall, the confusion matrix confirms that the model has learned robust and generalizable representations for all ten digit classes, with no single class exhibiting dramatically worse performance than the others.

### 6.4 Overfitting and Generalization

The gap between training accuracy (99.54%) and test accuracy (99.02%) is only 0.52 percentage points, which is remarkably small. This narrow gap indicates that the model has generalized well to unseen data and has not significantly overfit to the training set. Overfitting, which occurs when a model memorizes the training data at the expense of generalization, typically manifests as a large discrepancy between training and test performance. The absence of such a discrepancy here suggests that the model's capacity is well-matched to the complexity of the task, and that the combination of a compact architecture, a large training set (60,000 samples), and a limited number of training epochs (five) has acted as an effective implicit regularizer.

It is worth noting that no explicit regularization techniques, such as dropout, weight decay, or data augmentation, were employed in this project. The fact that the model still generalizes well without these techniques speaks to the favorable properties of the MNIST dataset (large training set relative to model complexity, limited intra-class variability compared to more challenging datasets like CIFAR-10 or ImageNet) and the inherent inductive biases of CNNs (weight sharing and local connectivity naturally limit model complexity).

---

## 7. Lessons Learned

This project provided several valuable technical and practical insights into the design, implementation, and evaluation of deep learning models for image classification.

The most striking technical insight is the sheer efficiency of CNNs on well-structured image classification tasks. The model achieved 99.02% test accuracy with only two convolutional layers, 128 hidden units in the fully connected layer, and five epochs of training on a CPU. This demonstrates that for problems where the data is clean, well-formatted, and of moderate complexity, even simple architectures can achieve near-state-of-the-art performance without extensive hyperparameter tuning or computational resources. The hierarchical feature learning mechanism of CNNs, which automatically discovers relevant patterns at multiple levels of abstraction, proved to be highly effective for digit recognition.

A second important insight relates to the role of preprocessing. The normalization step, using the dataset-specific mean and standard deviation, contributed to stable and fast convergence during training. Without normalization, the raw pixel values (ranging from 0 to 1 after tensor conversion) would have a skewed distribution (most pixels are zero, corresponding to the black background), which can lead to slower convergence and suboptimal gradient dynamics. The channel-wise normalization corrects for this imbalance and places the data in a regime where the Adam optimizer can operate most effectively.

Working with PyTorch also reinforced the importance of clean software engineering practices in machine learning projects. Separating the model definition, utility functions, and training script into distinct modules made the codebase easier to read, debug, and extend. Saving all metrics to a JSON file and all visualizations to image files ensured that the results are fully reproducible and can be inspected without re-running the training script.

One challenge encountered during the project was training on CPU rather than GPU. While the MNIST dataset is small enough that CPU training is feasible, the training process was noticeably slower than it would have been on a GPU. For larger datasets or more complex architectures, GPU acceleration would be essential to keep training times within practical limits.

Looking forward, several improvements could enhance this project. First, adding dropout layers between the fully connected layers would introduce explicit regularization and could potentially improve generalization, particularly if the model were trained for more epochs. Second, experimenting with learning rate scheduling (such as cosine annealing or step decay) could allow the model to converge to a sharper minimum, potentially pushing test accuracy even higher. Third, applying data augmentation techniques such as random rotation, slight translation, and elastic deformation would expose the model to a wider variety of handwriting styles and improve robustness. Finally, extending the project to more challenging datasets, such as Fashion-MNIST or CIFAR-10, would provide a more rigorous test of the architecture's capabilities and motivate the exploration of deeper and more sophisticated network designs.

---

## References

1. LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST Database of Handwritten Digits.* Available at: http://yann.lecun.com/exdb/mnist/
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
3. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.
4. Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

---
