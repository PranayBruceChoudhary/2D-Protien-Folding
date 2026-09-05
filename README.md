Review
Collapse All
### 1. Amino Acid Vocabulary
The input sequences consist of 20 standard amino acids plus an artificial padding character `'X'`:
$$\text{Alphabet} = \text{"ACDEFGHIKLMNPQRSTVWYX" } (21 \text{ tokens})$$
### 2. Sliding Window Technique
Secondary structure is determined by local interactions among neighboring amino acids. We use a **sliding window of size $W = 13$**:
- **Padding**: For a window of size 13, $\text{pad\_length} = 13 // 2 = 6$ characters of `'X'` are padded to the start and end of each protein sequence.
- **Window Extraction**: For each residue in the original protein sequence, a window of 13 characters centered at that residue is extracted.
- **Target Assignment**: The target label $Y$ is the 1-character secondary structure tag (`C`, `E`, or `H`) corresponding to the central residue.
### 3. One-Hot Vector Encoding
Each 13-character window is converted into a 1D binary vector for neural network ingestion:
- Each character is encoded as a 21-dimensional one-hot vector.
- A 13-character window is flattened into an input vector of size:
  $$\text{Input Dimension} = 13 \text{ positions} \times 21 \text{ amino acids} = 273 \text{ numerical features}$$
Target shapes are encoded into class indices:
$$\text{Coil } ('C') \rightarrow 0, \quad \text{Strand/Sheet } ('E') \rightarrow 1, \quad \text{Helix } ('H') \rightarrow 2$$
---
## 🤖 AI Model & Training: Linear Layer Network (`MiniFoldFFNN`)
### Model Architecture
The current model, `MiniFoldFFNN`, is implemented as a Feed-Forward Neural Network using PyTorch `nn.Module`:
```
Input (273 features)
       │
  ┌────▼────┐
  │Linear 1 │  --> Linear(in_features=273, out_features=256)
  └────┬────┘
       │
  ┌────▼────┐
  │  ReLU   │  --> Non-linear Activation
  └────┬────┘
       │
  ┌────▼────┐
  │Linear 2 │  --> Linear(in_features=256, out_features=3)
  └────┬────┘
       │
Output (3 Logits for C, E, H)
```
#### PyTorch Implementation Summary
```python
class MiniFoldFFNN(nn.Module):
    def __init__(self, input_size=273, hidden_size=256, output_size=3):
        super(MiniFoldFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```
### Training Setup
- **Loss Function**: `nn.CrossEntropyLoss()` (Categorical Cross-Entropy for 3-class classification).
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=0.001)`.
- **Batching**: `TensorDataset` and `DataLoader` with mini-batch size of 32 and random shuffling.
- **Inference Pipeline**: `predict_protein_structure()` handles end-to-end text sequence padding, windowing, encoding, forward pass execution, and class decoding back to structure strings.
---
## 🚀 Next Steps: Evolving to Convolutional Linear Layers (1D CNN)
While the Linear Layer model (`MiniFoldFFNN`) achieves initial structure predictions, fully-connected architectures have key limitations for biological sequence modeling:

### Limitations of Linear Layers
1. **Loss of Spatial Grid Structure**: Flattening the $13 \times 21$ window matrix into a 273-length 1D vector treats input indices as independent features, losing 2D positional relationship information.
2. **Parameter Growth & Overfitting**: Fully connected layers connect every input element to every neuron, leading to higher parameter counts without weight sharing.
3. **No Translation Invariance**: A motif (e.g. hydrophobic pattern) learned at positions 1-3 in the window is not automatically recognized if it appears at positions 8-10.
---
### Proposed Convolutional Architecture (`MiniFoldCNN`)
The next phase of the project will replace or augment the initial dense layers with **1D Convolutional Layers (`nn.Conv1d`)** combined with Linear classification heads.
```
Input Matrix (Batch, 21 channels, 13 sequence length)
                       │
             ┌─────────▼─────────┐
             │  nn.Conv1d Layer  │  (Kernels: 3, 5, or 7)
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │   BatchNorm1d     │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │       ReLU        │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │  Flatten / Pool   │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │  nn.Linear Head   │  (Dense output layer to 3 classes)
             └─────────┬─────────┘
                       │
          Predicted Structure (C, E, H)
```
### Key Advantages of Convolutional Layers for Protein Folding
1. **Local Biological Motif Detection**: Convolutions slide spatial filters (e.g., kernel size 3, 5, 7) over amino acid sequences. This directly captures local structural motifs, such as:
   - $\alpha$-helical hydrogen bonding patterns ($\approx 3.6$ residues per turn).
   - Alternating hydrophobic/hydrophilic patterns in $\beta$-sheets.
2. **Weight Sharing & Efficiency**: Convolutions reuse the same filter weights across all sequence positions, significantly reducing parameter counts while improving generalization.
3. **Translation Invariance**: Structural motifs are detected regardless of exact position within the sliding window.
4. **Deeper Context Integration**: Multiple stacked 1D Conv layers (or dilated convolutions) enable the network to expand its receptive field over longer amino acid context windows without parameter explosion.
---
## 🛠️ Requirements & Setup
### Prerequisites
- Python 3.9+
- PyTorch
- Pandas
- NumPy
### Installation
```bash
pip install torch pandas numpy
```
### Running the Notebook
Open [`Learn_Folding.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Learn_Folding.ipynb) in Jupyter Notebook or VS Code to train the model and test secondary structure predictions.
