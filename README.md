# 2D Protein Structure Prediction (MiniFold)

An AI/Deep Learning repository for predicting secondary protein structures (**Coil `C`**, **Beta-Sheet `E`**, and **Alpha-Helix `H`**) from primary amino acid sequences using PyTorch.

---

## 📌 Project Overview

Proteins are linear chains of amino acids that fold into complex 3D shapes to perform essential biological functions. Predicting secondary structure states from 1D primary sequences is a fundamental challenge in computational biology and structural bioinformatics.

This repository implements data preprocessing pipelines and deep learning architectures (**MiniFold**) in PyTorch. The project focuses primarily on a **1D Convolutional Neural Network (`MiniFoldCNN`)** for capturing spatial biological motifs, supported by a baseline **Feed-Forward Linear Network (`MiniFoldFFNN`)**.

---

## 📁 Repository Structure & File Inventory

| File | Description |
| :--- | :--- |
| 🚧 [`minifold_CNN.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/minifold_CNN.py) | **Primary Model (WIP)**: PyTorch script featuring end-to-end data loading, 3D tensor preparation (`[Batch, 21, 13]`), 1D Convolutional model (`MiniFoldCNN`), training loop, and inference testing. |
| 🚧 [`Minifold_CNN.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Minifold_CNN.ipynb) | **Primary Notebook (WIP)**: Interactive Jupyter Notebook for training, evaluating, and experimenting with the 1D CNN architecture. |
| [`minifold_linear.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/minifold_linear.py) | **Baseline Model**: PyTorch script implementing the fully-connected linear layer architecture (`MiniFoldFFNN`), one-hot vector encoding, training, and inference. |
| [`Minifold_Linear.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Minifold_Linear.ipynb) | **Baseline Notebook**: Interactive notebook for training and evaluating the baseline linear layer neural network. |
| [`RS126.data.txt`](file:///c:/Users/mannu/2D%20Protien%20Folding/RS126.data.txt) | Benchmark biological dataset containing paired lines of primary amino acid sequences and ground-truth secondary structure labels. |
| [`data_prep.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/data_prep.py) | Utility script parsing raw protein sequences and structure tags into structured Pandas DataFrames. |
| [`sliding_window.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/sliding_window.py) | Standalone demonstration of sequence padding and 13-residue sliding window generation. |

---

## 🧬 Data Pipeline & Feature Representation

### 1. Vocabulary & Alphabet
Input protein chains are parsed using a 21-token alphabet consisting of the 20 standard amino acids plus an artificial edge-padding character `'X'`:
$$\text{Alphabet} = \text{"ACDEFGHIKLMNPQRSTVWYX" } (21 \text{ tokens})$$

### 2. Sliding Window Technique
Local residue interactions determine secondary structure formation. We slice sequences using a **sliding window of size $W = 13$**:
- **Padding**: $\text{pad\_length} = 13 // 2 = 6$ characters of `'X'` prepended and appended to each protein sequence.
- **Window Extraction**: For each residue, a 13-character window centered at that position is extracted.
- **Target Assignment**: The target label $Y$ is the secondary structure character (`C`, `E`, or `H`) at the window's central position.

### 3. Tensor Formatting
- **CNN 3D Grid Format (`minifold_CNN.py`)**: One-hot encoded windows are shaped into 3D tensors `[Batch, Channels=21, Sequence_Length=13]` for 1D convolutions.
- **Linear 1D Vector Format (`minifold_linear.py`)**: One-hot windows are flattened into a 1D vector of size $13 \times 21 = 273$ numerical features.
- **Class Encoding**: 
  $$\text{Coil } ('C') \rightarrow 0, \quad \text{Strand/Sheet } ('E') \rightarrow 1, \quad \text{Helix } ('H') \rightarrow 2$$

---

## 🚧 1. Primary Model: Convolutional Neural Network (`MiniFoldCNN`)

> [!IMPORTANT]
> **Status**: **Work in Progress (Active Prototyping)**  
> The 1D Convolutional Neural Network (`MiniFoldCNN`) is the central architecture under active development in [`minifold_CNN.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/minifold_CNN.py) and [`Minifold_CNN.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Minifold_CNN.ipynb).

### Model Architecture
`MiniFoldCNN` processes inputs as a 3D spatial grid `[Batch, 21 channels, 13 positions]` to learn local spatial relationships between adjacent amino acids:

```
Input Grid Tensor [Batch, 21 Channels, 13 Window Length]
                          │
                ┌─────────▼─────────┐
                │  nn.Conv1d (L1)   │  --> in=21, out=64, kernel=3, padding=1
                └─────────┬─────────┘
                │ ReLU & Dropout(0.3)
                ┌─────────▼─────────┐
                │  nn.Conv1d (L2)   │  --> in=64, out=32, kernel=3, padding=1
                └─────────┬─────────┘
                │ ReLU
                ┌─────────▼─────────┐
                │     Flatten       │  --> Reshape [Batch, 32 * 13] -> [Batch, 416]
                └─────────┬─────────┘
                ┌─────────▼─────────┐
                │  nn.Linear Head   │  --> in=416, out=3
                └─────────┬─────────┘
                          │
           Raw Logits Output [Batch, 3] (C, E, H)
```

#### PyTorch Implementation (`minifold_CNN.py`)
```python
class MiniFoldCNN(nn.Module):
    def __init__(self, vocab_size=21, window_size=13, output_size=3):
        super(MiniFoldCNN, self).__init__()
        
        # Conv Layer 1: Slides 3-char filter across sequence
        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Conv Layer 2: Deeper biological motif extraction
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        
        # Linear Head: Flattened 3D feature grid -> 3 structure classes
        self.fc = nn.Linear(32 * window_size, output_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)  # Flatten [Batch, 32, 13] -> [Batch, 416]
        out = self.fc(x)
        return out
```

### Why Convolutional Layers for Protein Sequences?

1. **Local Biological Motif Extraction**: Convolutions slide learnable spatial kernels (kernel size 3) over sequence windows. This enables the network to directly learn recurring biological patterns, such as:
   - $\alpha$-helical hydrogen bonding turns ($\approx 3.6$ residues per turn).
   - Alternating hydrophobic/hydrophilic residue pairs in $\beta$-sheets.
2. **Translation Invariance**: Structural motifs are detected regardless of their exact position within the sliding window.
3. **Parameter Efficiency**: Filter weights are shared across spatial positions, preventing parameter explosion while improving generalization over raw linear projections.

---

## 🤖 2. Baseline Model: Linear Layer Network (`MiniFoldFFNN`)

> [!NOTE]
> **Status**: **Baseline Reference Implementation**  
> The fully-connected Feed-Forward Neural Network (`MiniFoldFFNN`) in [`minifold_linear.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/minifold_linear.py) serves as a baseline model for comparative performance analysis.

### Model Architecture
`MiniFoldFFNN` flattens the $13 \times 21$ window matrix into a 1D vector of 273 features:

```
Input (273 Flattened Features)
              │
        ┌─────▼─────┐
        │ Linear 1  │  --> Linear(in_features=273, out_features=128)
        └─────┬─────┘
        │ ReLU & Dropout(0.3)
        ┌─────▼─────┐
        │ Linear 2  │  --> Linear(in_features=128, out_features=3)
        └─────┬─────┘
              │
Output (3 Logits for C, E, H)
```

#### PyTorch Implementation (`minifold_linear.py`)
```python
class MiniFoldFFNN(nn.Module):
    def __init__(self, input_size=273, hidden_size=128, output_size=3):
        super(MiniFoldFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
```

---

## ⚙️ Training & Optimization Setup

Both models share a common training framework:
- **Loss Function**: `nn.CrossEntropyLoss()` (Categorical Cross-Entropy for discrete classes `C`, `E`, `H`).
- **Optimizer**: `torch.optim.Adam` with learning rate $\alpha = 0.001$.
- **Batching**: PyTorch `TensorDataset` and `DataLoader` (batch size 64 with shuffle).
- **Inference**: Custom sequence prediction functions (`predict_protein_structure_cnn` and `predict_protein_structure`) slide window representations across test sequences, run evaluation passes (`model.eval()`), and map argmax predictions back to structure strings.

---

## 🛠️ Installation & Execution

### Prerequisites
```bash
pip install torch pandas numpy
```

### Running Scripts
- **Train & Evaluate CNN Model (WIP)**:
  ```bash
  python minifold_CNN.py
  ```
- **Train & Evaluate Baseline Linear Model**:
  ```bash
  python minifold_linear.py
  ```
- Alternatively, launch [`Minifold_CNN.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Minifold_CNN.ipynb) or [`Minifold_Linear.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Minifold_Linear.ipynb) in Jupyter Notebook or VS Code.
