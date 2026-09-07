import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. DATA PREPARATION FUNCTIONS
# ==========================================

def load_and_parse_data(file_path):
    """Reads alternating lines of sequences and structures from the text file."""
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
    """Applies the sliding window and padding across protein sequences."""
    print(f"Generating sliding windows (using first {limit} proteins)...")
    pad_length = window_size // 2
    X_data = []
    Y_data = []
    
    # Process up to 'limit' proteins for fast prototyping (set limit=None for all)
    seq_subset = sequences[:limit] if limit else sequences
    struct_subset = structures[:limit] if limit else structures
    
    for seq, struct in zip(seq_subset, struct_subset):
        padded_seq = ("X" * pad_length) + seq + ("X" * pad_length)
        for j in range(len(seq)):
            X_data.append(padded_seq[j : j + window_size])
            Y_data.append(struct[j])
            
    print(f"Generated {len(X_data)} total training windows.")
    return X_data, Y_data


def prepare_cnn_tensors(X_data, Y_data, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """
    One-hot encodes X and reshapes into 3D format: [Batch, Channels, Length]
    Converts Y target letters (C, E, H) into integer labels (0, 1, 2).
    """
    print("Formatting tensors for CNN (3D Grid)...")
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    vocab_size = len(alphabet) # 21
    
    # 1. One-hot encode X into flat matrix: [Total_Rows, 13 * 21]
    X_flat = torch.zeros(len(X_data), window_size * vocab_size)
    for row_idx, window in enumerate(X_data):
        for char_idx, char in enumerate(window):
            if char in char_to_idx:
                col_idx = (char_idx * vocab_size) + char_to_idx[char]
                X_flat[row_idx, col_idx] = 1.0
                
    # 2. Reshape and Transpose to 3D: [Total_Rows, 21, 13]
    X_cnn = X_flat.view(-1, window_size, vocab_size).transpose(1, 2)
    
    # 3. Convert Y to integer labels
    shape_mapping = {'C': 0, 'E': 1, 'H': 2}
    Y_ints = [shape_mapping[shape] for shape in Y_data]
    Y_tensor = torch.tensor(Y_ints, dtype=torch.long)
    
    print(f"X_cnn shape: {X_cnn.shape}  [Batch, Channels, Length]")
    print(f"Y_tensor shape: {Y_tensor.shape}")
    return X_cnn, Y_tensor


# ==========================================
# 2. CNN MODEL ARCHITECTURE
# ==========================================

class MiniFoldCNN(nn.Module):
    def __init__(self, vocab_size=21, window_size=13, output_size=3):
        super(MiniFoldCNN, self).__init__()
        
        # Conv Layer 1: Slides 3-char filter across sequence
        self.conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Conv Layer 2: Deeper biological motif detection
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        
        # Final Classification Layer: Flattened 3D grid -> 3 classes
        self.fc = nn.Linear(32 * window_size, output_size)
        
    def forward(self, x):
        # Input shape: [Batch, 21, 13]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        # Flatten [Batch, 32, 13] -> [Batch, 416]
        x = x.view(x.size(0), -1)
        
        # Final Logits -> [Batch, 3]
        out = self.fc(x)
        return out


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================

def train_cnn_model(model, train_loader, num_epochs=10, lr=0.001):
    """Runs the training loop with Backpropagation and Loss calculation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- STARTING CNN TRAINING ---")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            # 1. Forward Pass
            outputs = model(batch_X)
            
            # 2. Loss Calculation
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


# ==========================================
# 4. INFERENCE FUNCTION
# ==========================================

def predict_protein_structure_cnn(protein_seq, trained_model, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """Takes a raw protein sequence string and predicts its secondary structure string."""
    trained_model.eval()
    
    pad_length = window_size // 2
    padded_seq = ("X" * pad_length) + protein_seq + ("X" * pad_length)
    
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    vocab_size = len(alphabet)
    protein_len = len(protein_seq)
    
    # 1. Build flat matrix: [Protein_Length, 273]
    X_flat = torch.zeros(protein_len, window_size * vocab_size)
    for i in range(protein_len):
        window = padded_seq[i : i + window_size]
        for char_idx, char in enumerate(window):
            if char in char_to_idx:
                col_idx = (char_idx * vocab_size) + char_to_idx[char]
                X_flat[i, col_idx] = 1.0
                
    # 2. Reshape to CNN 3D Tensor: [Protein_Length, 21, 13]
    X_inference = X_flat.view(-1, window_size, vocab_size).transpose(1, 2)
    
    # 3. Forward Pass
    with torch.no_grad():
        outputs = trained_model(X_inference)
        predicted_indices = torch.argmax(outputs, dim=1)
        
    # 4. Map integers back to letters
    int_to_shape = {0: 'C', 1: 'E', 2: 'H'}
    predicted_chars = [int_to_shape[int(idx.item())] for idx in predicted_indices]
    
    return "".join(predicted_chars)


# ==========================================
# 5. MAIN SCRIPT EXECUTION
# ==========================================

if __name__ == '__main__':
    # Configuration
    FILE_PATH = 'RS126.data.txt'  # <-- REPLACE WITH YOUR ACTUAL DATA FILE
    WINDOW_SIZE = 13
    ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    
    # Step 1: Ingest and Parse Data
    sequences, structures = load_and_parse_data(FILE_PATH)
    
    # Step 2: Slice Sliding Windows
    X_data, Y_data = create_sliding_windows(sequences, structures, window_size=WINDOW_SIZE, limit=50)
    
    # Step 3: Prepare PyTorch 3D Tensors
    X_cnn, Y_tensor = prepare_cnn_tensors(X_data, Y_data, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    # Step 4: Create DataLoader for mini-batches
    dataset = TensorDataset(X_cnn, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Step 5: Initialize Model
    model = MiniFoldCNN(vocab_size=len(ALPHABET), window_size=WINDOW_SIZE, output_size=3)
    
    # Step 6: Train Model
    trained_model = train_cnn_model(model, train_loader, num_epochs=EPOCHS, lr=LR)
    
    # Step 7: Test Inference
    test_input = "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"
    expected_output = "CCCCCCCCHHHHHHHHHHHHHHCECCCCCC"
    
    prediction = predict_protein_structure_cnn(test_input, trained_model, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    # Calculate simple accuracy on the test sequence
    matches = sum(1 for p, e in zip(prediction, expected_output) if p == e)
    accuracy = (matches / len(expected_output)) * 100
    
    print("--- CNN INFERENCE TEST ---")
    print(f"Input:    {test_input}")
    print(f"Expected: {expected_output}")
    print(f"CNN Pred: {prediction}")
    print(f"Accuracy: {accuracy:.1f}% ({matches}/{len(expected_output)} correct amino acids)")