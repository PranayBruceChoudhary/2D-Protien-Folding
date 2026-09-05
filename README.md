
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
