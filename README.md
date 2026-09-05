# 2D Protein Structure Prediction (MiniFold)
An AI/Deep Learning repository for predicting secondary protein structures (**Coil `C`**, **Beta-Sheet `E`**, and **Alpha-Helix `H`**) from primary amino acid sequences using PyTorch.
---
## 📌 Project Overview
Proteins are sequence chains of amino acids that fold into complex 3D shapes to perform biological functions. Predicting secondary structure from the 1D primary sequence is a fundamental step in computational biology and protein folding.
This project implements a complete data pipeline and neural network model (**MiniFoldFFNN**) using PyTorch. The current implementation uses a **sliding window approach** combined with **one-hot encoding** and a **Linear Fully-Connected Neural Network (FFNN)** to classify each amino acid residue into its secondary structure state (`C`, `E`, or `H`).
---
## 📁 Repository Structure & File Inventory
| File | Description |
| :--- | :--- |
| [`RS126.data.txt`](file:///c:/Users/mannu/2D%20Protien%20Folding/RS126.data.txt) | Benchmark biological dataset containing paired lines of primary amino acid sequences and ground-truth secondary structure sequences. |
| [`sliding_window.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/sliding_window.py) | Standalone script demonstrating sequence padding and the 13-residue sliding window extraction algorithm on sample sequence data. |
| [`data_prep.py`](file:///c:/Users/mannu/2D%20Protien%20Folding/data_prep.py) | Data parsing script that extracts amino acid sequences and structure tags from `RS126.data.txt` into a structured Pandas DataFrame. |
| [`Learn_Folding.ipynb`](file:///c:/Users/mannu/2D%20Protien%20Folding/Learn_Folding.ipynb) | Main PyTorch notebook containing end-to-end data preprocessing, one-hot encoding, model definition (`MiniFoldFFNN`), training loop, and inference functions. |
---
## 🧬 Data Pipeline & Feature Representation
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
