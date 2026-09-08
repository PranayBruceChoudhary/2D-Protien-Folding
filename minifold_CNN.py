import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =========================================================
# 1. DATA PREPARATION FUNCTIONS
# =========================================================

def load_and_parse_data(file_path):
    """Reads alternating lines of sequences and structures from the raw text file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    sequences = []
    structures = []
    
    # Step through lines 2 at a time (Seq, Struct, Seq, Struct...)
    for i in range(0, len(lines) - 1, 2):
        seq = lines[i].strip()
        struct = lines[i+1].strip()
        if len(seq) == len(struct) and len(seq) > 0:
            sequences.append(seq)
            structures.append(struct)
            
    print(f"Successfully loaded {len(sequences)} protein chains.")
    return sequences, structures


def create_sliding_windows(sequences, structures, window_size=13, limit=50):
    """Applies the sliding window and edge padding across protein sequences."""
    print(f"Generating sliding windows (using first {limit} proteins)...")
    pad_length = window_size // 2
    X_data = []
    Y_data = []
    
    seq_subset = sequences[:limit] if limit else sequences
    struct_subset = structures[:limit] if limit else structures
    
    for seq, struct in zip(seq_subset, struct_subset):
        padded_seq = ("X" * pad_length) + seq + ("X" * pad_length)
        for j in range(len(seq)):
            X_data.append(padded_seq[j : j + window_size])
            Y_data.append(struct[j])
            
    print(f"Generated {len(X_data)} total training windows.")
    return X_data, Y_data


def prepare_embedding_tensors(X_data, Y_data, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """
    Converts 13-character string windows into 2D integer tensors [Total_Rows, 13] for nn.Embedding.
    Converts Y target letters (C, E, H) into integer labels (0, 1, 2).
    """
    print("Formatting tensors for Embedding Layer (Integer Tokens)...")
    char_to_index = {char: idx for idx, char in enumerate(alphabet)}
    
    # 1. Store raw integer IDs directly: Shape [Total_Rows, 13]
    X_ints = torch.zeros(len(X_data), window_size, dtype=torch.long)
    for row_idx, window in enumerate(X_data):
        for char_idx, char in enumerate(window):
            if char in char_to_index:
                X_ints[row_idx, char_idx] = char_to_index[char]
                
    # 2. Target labels (C, E, H -> 0, 1, 2)
    shape_mapping = {'C': 0, 'E': 1, 'H': 2}
    Y_ints = [shape_mapping[shape] for shape in Y_data]
    Y_tensor = torch.tensor(Y_ints, dtype=torch.long)
    
    print(f"X_ints shape: {X_ints.shape}  [Total_Rows, 13]")
    print(f"Y_tensor shape: {Y_tensor.shape}")
    return X_ints, Y_tensor


# =========================================================
# 2. NEURAL NETWORK ARCHITECTURE
# =========================================================

class LearningBlock(nn.Module):
    """
    A single reusable learning block consisting of:
    Linear Layer (256 -> 256) -> BatchNorm1d -> ReLU -> Dropout
    """
    def __init__(self, input_channels=256, dropout_rate=0.3):
        super(LearningBlock, self).__init__()
        self.linear1 = nn.Linear(input_channels, input_channels)
        self.batch_norm1 = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MiniFoldCNN(nn.Module):
    """
    Timeline:
    Integer Tokens -> Embedding Layer -> Conv1D Feature Extractor -> 
    Bridge Layer (1664 -> 256) -> Dynamic LearningBlocks (SequenceCount) -> Linear Classifier (256 -> 3)
    """
    def __init__(self, vocab_size=21, input_channels=32, window_size=13, output_size=3, SequenceCount=2, hidden_dim=256):
        super(MiniFoldCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_channels)
        
        # Dimensions
        conv_out_channels = 128
        flattened_dim = conv_out_channels * window_size  # 128 * 13 = 1,664
        
        # 2. Sequential Pipeline
        self.layers = nn.Sequential(
            # Feature Extractor (Conv1D)
            nn.Conv1d(in_channels=input_channels, out_channels=conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Bridge 3D Conv grid -> 2D Flat vector [Batch, 1664]
            nn.Flatten(),
            
            # Projection Bridge: Shrink from 1,664 down to 256 for the LearningBlocks
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            
            # Stacking your LearningBlocks dynamically! (256 -> 256)
            *[LearningBlock(input_channels=hidden_dim, dropout_rate=0.3) for _ in range(SequenceCount)],
            
            # Final Classification Head (256 -> 3 classes: C, E, H)
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        # x shape entering: [Batch, 13] (raw integers)
        x = self.embedding(x)       # -> [Batch, 13, 32]
        x = x.transpose(1, 2)       # -> [Batch, 32, 13] (Batch, Channels, Length)
        x = self.layers(x)          # -> [Batch, 3]
        return x


# =========================================================
# 3. TRAINING FUNCTION
# =========================================================

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    """Executes the training loop with Backpropagation, Loss tracking, and Weight updates."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- STARTING MODEL TRAINING ---")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            # 1. Forward Pass
            outputs = model(batch_X)
            
            # 2. Calculate Loss
            loss = criterion(outputs, batch_Y)
            
            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Average Loss: {avg_loss:.4f}")
        
    print("--- TRAINING COMPLETE ---\n")
    return model


# =========================================================
# 4. INFERENCE FUNCTION
# =========================================================

def predict_protein_structure(protein_seq, trained_model, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """Takes a full protein sequence and predicts its secondary structure string."""
    trained_model.eval() # Turn off Dropout
    
    pad_length = window_size // 2
    padded_seq = ("X" * pad_length) + protein_seq + ("X" * pad_length)
    
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    protein_len = len(protein_seq)
    
    # 1. Build 2D integer tensor: [Protein_Length, 13]
    X_test_ints = torch.zeros(protein_len, window_size, dtype=torch.long)
    for i in range(protein_len):
        window = padded_seq[i : i + window_size]
        for char_idx, char in enumerate(window):
            if char in char_to_idx:
                X_test_ints[i, char_idx] = char_to_idx[char]
                
    # 2. Run Forward Pass
    with torch.no_grad():
        outputs = trained_model(X_test_ints) # Shape: [Length, 3]
        predicted_indices = torch.argmax(outputs, dim=1)
        
    # 3. Translate integers back to letters
    int_to_shape = {0: 'C', 1: 'E', 2: 'H'}
    predicted_chars = [int_to_shape[int(idx.item())] for idx in predicted_indices]
    
    return "".join(predicted_chars)


# =========================================================
# 5. MAIN SCRIPT EXECUTION
# =========================================================

if __name__ == '__main__':
    # Configuration Hyperparameters
    FILE_PATH = 'RS126.data.txt'  # <-- REPLACE WITH YOUR ACTUAL FILE NAME
    WINDOW_SIZE = 13
    ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    EMBED_DIM = 32
    SEQUENCE_COUNT = 3  # Try changing this to 1, 3, or 5 to see how depth affects learning!
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    
    # Step 1: Load and parse biological data
    sequences, structures = load_and_parse_data(FILE_PATH)
    
    # Step 2: Slice Sliding Windows (first 50 proteins for rapid training)
    X_data, Y_data = create_sliding_windows(sequences, structures, window_size=WINDOW_SIZE, limit=50)
    
    # Step 3: Prepare Integer Tensors for Embedding Layer
    X_ints, Y_tensor = prepare_embedding_tensors(X_data, Y_data, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    # Step 4: Setup DataLoader for Mini-Batching
    dataset = TensorDataset(X_ints, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Step 5: Initialize Dynamic MiniFold CNN Model
    model = MiniFoldCNN(
        vocab_size=len(ALPHABET),
        input_channels=EMBED_DIM,
        window_size=WINDOW_SIZE,
        output_size=3,
        SequenceCount=SEQUENCE_COUNT,
        hidden_dim=HIDDEN_DIM
    )
    
    print("Model initialized successfully! Model architecture summary:")
    print(model)
    
    # Step 6: Train the Model
    trained_model = train_model(model, train_loader, num_epochs=EPOCHS, lr=LR)
    
    # Step 7: Run Inference Test
    test_input = "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"
    expected_output = "CCCCCCCCHHHHHHHHHHHHHHCECCCCCC"
    
    prediction = predict_protein_structure(test_input, trained_model, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    matches = sum(1 for p, e in zip(prediction, expected_output) if p == e)
    accuracy = (matches / len(expected_output)) * 100
    
    print("--- INFERENCE TEST RESULTS ---")
    print(f"Input:    {test_input}")
    print(f"Expected: {expected_output}")
    print(f"Pred:     {prediction}")
    print(f"Accuracy: {accuracy:.1f}% ({matches}/{len(expected_output)} correct amino acids)")