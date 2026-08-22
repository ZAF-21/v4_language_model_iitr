# Introduction to PyTorch — End-to-End Notebook Flow

```
START — begin with ordinary Python numbers and end with a CNN that classifies photographs
  │
  ▼
[P0] SETUP — make every later experiment reproducible and hardware-agnostic
  ├── import the five pieces used throughout
  │     torch · torch.nn · torch.nn.functional · torch.optim · DataLoader
  ├── seed Python, NumPy, PyTorch and CUDA RNGs
  │     └── reproducible within the limits of hardware and library versions
  ├── choose one device: MPS → CUDA → CPU
  │     └── model and data must always live on the same device
  └── create data/ — torchvision downloads once, then reuses the cache
  │
  ▼
[P1] TENSORS — NumPy-style arrays with two extra powers
  ├── familiar core
  │     ├── shapes, indexing, arithmetic, sum(), mean()
  │     ├── torch.tensor / zeros / randn
  │     └── NumPy ↔ PyTorch conversion
  │           caveat: NumPy commonly gives float64; models expect float32
  │
  ├── extra power 1: hardware placement
  │     └── x = x.to(device)
  │           ├── .to() returns a moved tensor; it does not mutate x
  │           └── CPU tensor + GPU tensor → device-mismatch error
  │
  ├── extra power 2: operation history
  │     └── requires_grad=True records how a result was produced
  │           → the computational graph used by autograd
  │
  └── operations needed later
        ├── view(N, -1) ───────────── flatten each sample
        ├── argmax(dim=1) ─────────── logits → predicted class index
        ├── float32 data / int64 labels
        ├── item() ────────────────── scalar tensor → Python number
        └── detach().cpu().numpy() ── safe hand-off to NumPy/plots
  │
  ▼
[P2] AUTOGRAD — determine how every tracked number affects one scalar loss
  │
  ├── dummy forward pass: y = (x - 3)^2 + z, with x=5 and z=2
  │     └── y=6 carries grad_fn links back through add → power → subtract
  │
  ├── y.backward() walks that graph in reverse using the chain rule
  │     ├── x.grad = 2(x-3) = 4
  │     └── z.grad = 1
  │           read a gradient as: “which direction increases the result?”
  │
  ├── rules that explain later training code
  │     ├── backward starts from a scalar
  │     │     vector output → reduce with mean/sum or supply a gradient
  │     ├── gradients accumulate rather than overwrite
  │     │     → clear them before every new backward pass
  │     ├── torch.no_grad() disables graph construction for evaluation
  │     └── detach() cuts one tensor away from its graph
  │
  ├── use the gradient to minimise y = (x - 3)^2
  │     └── x=10 → forward → backward → x -= lr*grad → clear → repeat
  │           after 30 steps with lr=0.1: x ≈ 3.0087
  │
  └── replace manual bookkeeping with an optimizer
        ├── optimizer.zero_grad() ── clear old gradients
        └── optimizer.step() ────── update every tensor it owns
  │
  ▼
[P3] nn.Module — scale the one-number lesson to a network of parameters
  │
  ├── nn.Linear(in, out)
  │     └── y = xWᵀ + b; weights and bias are registered automatically
  ├── insert a non-linearity between Linear layers
  │     └── without ReLU/GELU/etc., many Linear layers collapse into one
  ├── define a model
  │     ├── __init__ creates and registers layers on self
  │     ├── forward describes the batch computation
  │     └── call model(x), not model.forward(x)
  ├── model.parameters()
  │     └── the common list used by .to(device), autograd and the optimizer
  └── keep the three responsibilities separate
        │
        │  input → MODEL → raw logits → LOSS → one scalar
        │                   nn.Module     CrossEntropyLoss
        │                                      │
        │                                      ▼ backward()
        │                                   gradients
        │                                      │
        └──────────── updated weights ◀── OPTIMIZER
  
        CrossEntropyLoss contract:
        ├── logits: (N, number_of_classes), with NO model-side softmax
        ├── labels: (N,), integer class indices, dtype int64
        └── fresh random model loss should be about ln(number_of_classes)
              MNIST: ln(10) ≈ 2.303 · CIFAR-100: ln(100) ≈ 4.605
  │
  ▼
[P4] DATA PIPELINE — turn stored images into shuffled training batches
  │
  ├── PIL image
  │     ▼ ToTensor()
  │   HWC uint8 [0,255] → CHW float32 [0,1]
  │     ▼ Normalize(mean, std)
  │   centred/scaled tensor → easier gradient optimisation
  │
  ├── Dataset
  │     └── sample i → (image, label); only one sample must be loaded at a time
  └── DataLoader
        ├── gathers samples into batches of 128
        ├── train: shuffle=True
        ├── test:  shuffle=False
        └── emits the PyTorch vision convention
              images: (N, C, H, W) · labels: (N,)
  │
  ▼
══════════════════════ the reusable train/evaluate engine ══════════════════════
  │
  ├── TRAIN one epoch
  │     model.train()
  │     for x, y in train_loader:
  │       1. x, y = x.to(device), y.to(device)
  │       2. logits = model(x)                 forward
  │       3. loss = criterion(logits, y)       one scalar error
  │       4. optimizer.zero_grad()             remove stale gradients
  │       5. loss.backward()                   autograd fills every .grad
  │       6. optimizer.step()                  update every parameter
  │
  └── EVALUATE one epoch
        model.eval()
        with torch.no_grad():
          forward only → collect loss, accuracy and predictions

     The engine does not care whether model is an MLP, CNN or something larger.
     Parts 5–7 keep this engine essentially fixed and change the data/model.
  │
  ▼
[P5] EXPERIMENT 1 — MNIST + MLP: establish that the whole pipeline works
  │
  ├── data: 60,000 train + 10,000 test
  │     └── batch shape (128, 1, 28, 28); 10 digit classes
  ├── MLP shape trace
  │     (N,1,28,28) → flatten (N,784) → 128 → ReLU → 64 → ReLU → 10 logits
  ├── 109,386 trainable parameters
  ├── preflight checks
  │     ├── output shape: (128,10)
  │     ├── initial loss: 2.312 ≈ ln(10)
  │     └── initial accuracy: near the 10% random baseline
  ├── train: SGD(lr=0.1), 8 epochs
  └── result: 97.71% test accuracy
        ├── plot train/test loss and accuracy to detect overfitting
        ├── classification report + confusion matrix for per-digit behaviour
        └── inspect misclassified images; the common error was 9 → 4
  │
  ▼
[P6] EXPERIMENT 2 — CIFAR-100 + the SAME MLP: expose an architectural failure
  │
  ├── harder data: 50,000 train + 10,000 test
  │     ├── colour images: (3,32,32) = 3,072 input values
  │     ├── 100 classes; only 500 train images per class
  │     └── random baseline: 1%
  ├── resize only the MLP endpoints
  │     (N,3,32,32) → flatten (N,3072) → 128 → 64 → 100 logits
  │     └── 408,100 parameters; first dense layer alone uses 393,344
  ├── preflight: (128,3,32,32) → (128,100), loss 4.606 ≈ ln(100)
  ├── train: Adam(lr=1e-3), 20 epochs
  └── result: about 22.8% test accuracy while train accuracy reaches 39.1%
        └── the widening train/test gap says memorisation, not good generalisation
  │
  ├── WHY did the MNIST architecture fail on photographs?
  │     ├── flattening destroys the 2-D neighbourhood structure
  │     ├── each location has unrelated weights → no built-in shift tolerance
  │     └── dense connections spend a separate parameter on every pixel-neuron pair
  │
  └── measure the position problem instead of merely claiming it
        ├── roll every test image sideways by 0,1,2,4,8 pixels
        ├── 0 px: 22.79% accuracy
        └── 8 px:  9.66% accuracy → retains only 42% of its unshifted score
              same objects, different alignment, sharply worse predictions
  │
  ▼
[P7] EXPERIMENT 3 — CIFAR-100 + CNN: build spatial structure into the model
  │
  ├── first understand convolution directly
  │     ├── slide one small kernel over local pixel neighbourhoods
  │     ├── multiply + sum at each location → one feature map
  │     └── hand-written edge, blur and sharpen kernels prove the operation
  │
  ├── why convolution answers the MLP’s three failures
  │     ├── locality ───────── kernels see neighbouring pixels
  │     ├── weight sharing ── the same detector works at every location
  │     └── parameter economy: Conv2d(3,32,3,padding=1) needs only 896 params
  │           (8,3,32,32) → (8,32,32,32)
  │
  ├── MaxPool2d(2,2)
  │     └── keeps each 2×2 maximum: halves H and W, preserves channels,
  │           reduces computation and adds some tolerance to small shifts
  │
  ├── CNN shape trace
  │     (N,  3,32,32)
  │       → conv + BN + ReLU + conv + BN + ReLU + pool
  │     (N, 32,16,16)
  │       → conv + BN + ReLU + conv + BN + ReLU + pool
  │     (N, 64, 8, 8)
  │       → conv + BN + ReLU + pool
  │     (N,128, 4, 4)
  │       → flatten (N,2048) → Dropout(0.3) → Linear → (N,100) logits
  │
  ├── BatchNorm and Dropout make mode switching real
  │     ├── train(): batch statistics + random dropout
  │     └── eval(): running statistics + dropout disabled
  │
  ├── train with the SAME engine: Adam(lr=1e-3), 20 epochs
  └── controlled verdict on the same CIFAR-100 data
        ├── MLP: 408,100 parameters → 22.75% test accuracy
        ├── CNN: 344,964 parameters → 54.97% test accuracy
        └── 85% as many parameters, about 2.4× the accuracy
  │
  ├── repeat the shift test
  │     ├── MLP at 8 px:  9.66% · retains 42% of its baseline
  │     └── CNN at 8 px: 30.52% · retains 55% of its baseline
  │           improvement is real, but convolution gives equivariance;
  │           pooling/aggregation provide only imperfect invariance
  │
  └── inspect what remains
        ├── calculate per-class accuracy: easiest vs hardest categories
        └── visualise first-layer feature maps
              → learned edge and colour-contrast detectors emerge from the loss
  │
  ▼
[P8] SAVE, RELOAD AND INFER — turn a notebook model into a reusable model
  │
  ├── state_dict()
  │     └── named tensors only; architecture remains in the Python class
  ├── save cnn_model.state_dict() → cifar100_cnn.pth (~1.4 MB)
  ├── reload safely
  │     1. construct the same CNN architecture
  │     2. load_state_dict(torch.load(..., weights_only=True))
  │     3. call eval()
  │     └── verify original and reloaded models emit identical logits
  ├── resumable checkpoint adds
  │     └── epoch + model state + optimizer state + recorded metric
  └── predict one image
        ├── transformed image: (3,32,32)
        ├── unsqueeze(0) → batch of one: (1,3,32,32)
        ├── eval() + no_grad() → logits: (1,100)
        ├── softmax at inference → human-readable probabilities
        └── topk(5) → ranked class predictions
  │
  ▼
END RESULT — what the notebook has actually established
  • Tensor holds and moves numbers; autograd computes their gradients.
  • nn.Module owns parameters; Dataset/DataLoader supplies batches.
  • One reusable forward → loss → zero → backward → step loop trains every model.
  • High MNIST accuracy does not prove an architecture understands images.
  • On CIFAR-100, preserving spatial structure with convolution matters more than
    giving a flattened MLP more parameters or more epochs.
  • A trained model becomes useful only after correct save/load and inference handling.

[P9] HABITS reinforced throughout the notebook
  • inspect the data and print one batch before training
  • trace every important tensor shape instead of guessing
  • check fresh loss ≈ ln(number_of_classes)
  • emit logits; do not place softmax before CrossEntropyLoss
  • keep model and data on the same device
  • clear gradients before backward; update only after backward
  • use train() for training and eval() + no_grad() for evaluation
  • compare train and test curves, then inspect actual errors
  • test architectural claims with controlled measurements (same data and loop)
  • save state_dict; reload into explicit code; verify outputs match
```

The central narrative is deliberately experimental: build the four PyTorch mechanisms,
prove the training loop on MNIST, watch the same MLP fail on CIFAR-100, diagnose why it
fails, and then show that a CNN fixes much of the problem without changing the training
engine.
