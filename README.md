# nn_cpp — Neural Network from Scratch in C++

A fully hand-written neural network implementation in C++ that trains on and classifies the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset. No external ML libraries are used — every component (forward propagation, backpropagation, activation functions, weight initialization) is implemented from first principles.

## Network Architecture

The network is a fully-connected (dense) feedforward neural network with four layers:

```
Input Layer (784) → Hidden Layer 1 (512) → Hidden Layer 2 (256) → Output Layer (10)
```

| Layer | Size | Activation | Purpose |
|---|---|---|---|
| Input | 784 neurons | Identity (pass-through) | One neuron per pixel of a 28×28 grayscale image |
| Hidden 1 | 512 neurons | ReLU | First feature extraction layer |
| Hidden 2 | 256 neurons | ReLU | Second feature extraction layer |
| Output | 10 neurons | Softmax | Probability distribution over digits 0–9 |

### Hyperparameters

All hyperparameters are defined in `include/NNParams.h`:

- **Learning Rate:** 0.01
- **Weight Initialization:** He-normal (Gaussian with σ = √(2 / fan_in))
- **Bias Initialization:** 0.0
- **Optimizer:** Vanilla Stochastic Gradient Descent (SGD)
- **Epochs:** 1 (single pass over the training set)

## How It Works

### Forward Propagation

1. **Input normalization** — Raw pixel values (0–255) are scaled to the [0, 1] range by dividing by 255.
2. **Weighted sum** — For each neuron in the next layer, a dot product of the previous layer's outputs and the connection weights is computed, plus a bias term: `z = Σ(wᵢ · xᵢ) + b`.
3. **Activation** — Hidden layers apply the ReLU function (`max(0, z)`). The output layer applies Softmax to produce a probability distribution over the 10 digit classes.

The function named `convolution` in `Calculations.cpp` is actually a **dense (fully-connected) layer computation** — it computes the weighted sum across all inputs for each output neuron. Despite the name, no convolutional filters or kernels are involved.

### Backpropagation

Gradients are computed layer-by-layer from output to input using the chain rule:

1. **Output layer delta** — The error signal at the output is `δ_out = predicted - target`, where target is a one-hot encoded vector of the true label.
2. **Hidden layer deltas** — Each hidden layer's delta is computed by propagating the downstream delta backward through the weights, multiplied by the derivative of ReLU.
3. **Weight & bias updates** — Weights and biases are updated using vanilla SGD: `w = w - lr × gradient`.

### Weight Initialization

Weights are initialized using **He-normal initialization**, which draws from a Gaussian distribution with mean 0 and standard deviation `√(2 / fan_in)`. This is the recommended strategy for networks using ReLU activations and helps prevent vanishing/exploding gradients during training.

## Project Structure

```
nn_cpp/
├── include/                  # Header files
│   ├── BackProp.h            # Backpropagation class declaration
│   ├── Calculations.h        # Activation functions & layer computation
│   ├── ForwardProp.h         # Forward propagation class declaration
│   ├── Network.h             # Main network orchestrator
│   ├── NNParams.h            # Hyperparameters (layer sizes, learning rate)
│   ├── Node.h                # Neuron class declaration
│   └── Test.h                # Testing/evaluation class declaration
├── src/                      # Source files
│   ├── BackProp.cpp          # Gradient computation and weight updates
│   ├── Calculations.cpp      # ReLU, Softmax, dense layer forward pass
│   ├── ForwardProp.cpp       # Feeds input through the network
│   ├── Network.cpp           # Training loop, MNIST file I/O, timing
│   ├── NNParams.cpp          # (Placeholder for namespace constants)
│   ├── Node.cpp              # Neuron: weights, bias, He initialization
│   ├── Test.cpp              # Evaluation loop on test set, accuracy calc
│   └── main.cpp              # Entry point
├── lib/                      # Compiled object files (.o)
├── bin/                      # Compiled binary
├── train-images.idx3-ubyte   # MNIST training images (60,000)
├── train-labels.idx1-ubyte   # MNIST training labels
├── t10k-images.idx3-ubyte    # MNIST test images (10,000)
├── t10k-labels.idx1-ubyte    # MNIST test labels
├── makefile                  # Build configuration
└── LICENSE                   # MIT License
```

### Class Responsibilities

| Class | File(s) | Responsibility |
|---|---|---|
| `Node` | Node.h / Node.cpp | Represents a single neuron. Stores input, output, bias, and outgoing weights. Handles He-normal weight initialization and inline ReLU during `setInput`. |
| `Calculations` | Calculations.h / Calculations.cpp | Provides ReLU activation, its derivative, Softmax, and the dense layer weighted-sum computation. |
| `ForwardProp` | ForwardProp.h / ForwardProp.cpp | Loads a single image into the input layer and runs a full forward pass through all layers. |
| `BackProp` | BackProp.h / BackProp.cpp | Computes deltas for each layer and updates all weights and biases using SGD. |
| `Network` | Network.h / Network.cpp | Top-level orchestrator. Initializes layers, reads MNIST binary files, runs the training loop (forward + backward for each image), and delegates testing. Measures and reports timing. |
| `Test` | Test.h / Test.cpp | Reads the MNIST test set, runs forward propagation on each image, compares predictions to labels, and reports accuracy. |
| `NNParams` | NNParams.h / NNParams.cpp | Compile-time constants namespace for layer sizes and learning rate. |

## Building and Running

### Prerequisites

- A C++ compiler supporting C++11 or later (e.g., `g++`)
- The MNIST dataset files (already included in the repository)

### Build

```bash
make compile
```

This compiles all source files into object files in `lib/` and links them into an executable at `bin/main`.

### Run

```bash
make run
```

Or build and run in one step:

```bash
make
```

### What to Expect

1. The program initializes the network with random weights.
2. It trains on all 60,000 MNIST training images in a single epoch, printing progress after each image.
3. After training, it evaluates on the 10,000 test images, printing per-image predictions and a running accuracy percentage.
4. Finally, it displays total training time, testing time, and combined time.

## Dataset

The project uses the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/), stored in the IDX binary format. The four required files are included directly in the repository:

| File | Contents | Count |
|---|---|---|
| `train-images.idx3-ubyte` | Training images (28×28 grayscale) | 60,000 |
| `train-labels.idx1-ubyte` | Training labels (0–9) | 60,000 |
| `t10k-images.idx3-ubyte` | Test images (28×28 grayscale) | 10,000 |
| `t10k-labels.idx1-ubyte` | Test labels (0–9) | 10,000 |

The IDX binary format is parsed manually in `Network.cpp` and `Test.cpp` by reading the magic number, dimensions, and raw pixel/label bytes.

## Implementation Details

### ReLU Activation

```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

Applied to hidden layers to introduce non-linearity. The derivative is used during backpropagation to gate the gradient flow.

### Softmax Output

The output layer uses Softmax to convert raw scores (logits) into a probability distribution:

```
softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)
```

The predicted class is the index with the highest probability (`argmax`).

### Training Pipeline (per image)

1. Read one 28×28 image and its label from the MNIST binary files.
2. Normalize pixel values to [0, 1].
3. Forward propagation through all layers.
4. Compute cross-entropy-style gradient at the output (predicted − one-hot target).
5. Backpropagate deltas through hidden layers.
6. Update all weights and biases with SGD.

## Potential Improvements

- **Mini-batch training** — Process multiple images before updating weights to stabilize gradients and enable parallelism.
- **Multiple epochs** — Train for more than one pass over the dataset to improve accuracy.
- **Learning rate scheduling** — Decay the learning rate over time for finer convergence.
- **Momentum / Adam optimizer** — Replace vanilla SGD with a more advanced optimizer.
- **Softmax numerical stability** — Subtract the max logit before exponentiation to avoid overflow.
- **Dropout / regularization** — Add dropout layers or L2 regularization to reduce overfitting.
- **Configurable architecture** — Allow layer sizes and depth to be set at runtime or via a config file.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Author

**Veysel Aras** — [GitHub](https://github.com/veyselaras)
