import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
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

def create_sliding_windows(sequences, structures, window_size=13):
    """Applies the sliding window and padding across all protein sequences."""
    print("Generating sliding windows...")
    pad_length = window_size // 2
    X_data = []
    Y_data = []
    
    # For testing/prototyping, let's process the first 50 proteins to keep it fast
    # (Remove [:50] later if you want to train on the whole dataset!)
    for seq, struct in zip(sequences[:50], structures[:50]):
        padded_seq = ("X" * pad_length) + seq + ("X" * pad_length)
        
        for j in range(len(seq)):
            window = padded_seq[j : j + window_size]
            target = struct[j]
            
            X_data.append(window)
            Y_data.append(target)
            
    print(f"Generated {len(X_data)} total training windows.")
    return X_data, Y_data

def one_hot_encode_x(X_data, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """Turns 13-character string windows into flat 2D PyTorch tensors of 1s and 0s."""
    print("One-hot encoding X inputs...")
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    vocab_size = len(alphabet)
    
    X_flat = torch.zeros(len(X_data), window_size * vocab_size)
    
    for row_idx, window in enumerate(X_data):
        for char_idx, char in enumerate(window):
            if char in char_to_idx:
                col_idx = (char_idx * vocab_size) + char_to_idx[char]
                X_flat[row_idx, col_idx] = 1.0
                
    return X_flat

def prepare_y_tensor(Y_data):
    """Converts target shape letters (C, E, H) into integer tensor labels (0, 1, 2)."""
    print("Converting Y targets to integers...")
    shape_mapping = {'C': 0, 'E': 1, 'H': 2}
    Y_ints = [shape_mapping[shape] for shape in Y_data]
    return torch.tensor(Y_ints, dtype=torch.long)


# ==========================================
# 2. NEURAL NETWORK MODEL CLASS
# ==========================================

class MiniFoldFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    """Handles the training loop, loss calculation, and backpropagation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- STARTING LINEAR MODEL TRAINING ---")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            # 1. Forward pass
            outputs = model(batch_X)
            
            # 2. Calculate loss
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

def predict_protein_structure(protein_seq, trained_model, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """Takes a full protein sequence and predicts its secondary structure string."""
    trained_model.eval()
    
    pad_length = window_size // 2
    padded_seq = ("X" * pad_length) + protein_seq + ("X" * pad_length)
    
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    vocab_size = len(alphabet)
    
    X_inference = torch.zeros(len(protein_seq), window_size * vocab_size)
    
    for i in range(len(protein_seq)):
        window = padded_seq[i : i + window_size]
        for char_idx, char in enumerate(window):
            if char in char_to_idx:
                col_idx = (char_idx * vocab_size) + char_to_idx[char]
                X_inference[i, col_idx] = 1.0
                
    with torch.no_grad():
        outputs = trained_model(X_inference)
        predicted_indices = torch.argmax(outputs, dim=1)
        
    int_to_shape = {0: 'C', 1: 'E', 2: 'H'}
    predicted_chars = [int_to_shape[int(idx.item())] for idx in predicted_indices]
    
    return "".join(predicted_chars)


# ==========================================
# 5. MAIN SCRIPT EXECUTION
# ==========================================

if __name__ == '__main__':
    # Configuration
    FILE_PATH = 'RS126.data.txt' # <-- REPLACE WITH YOUR ACTUAL FILE NAME
    WINDOW_SIZE = 13
    ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    
    # Step 1: Load and parse data
    sequences, structures = load_and_parse_data(FILE_PATH)
    
    # Step 2: Create windows
    X_data, Y_data = create_sliding_windows(sequences, structures, window_size=WINDOW_SIZE)
    
    # Step 3: Convert to Tensors
    X_tensor = one_hot_encode_x(X_data, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    Y_tensor = prepare_y_tensor(Y_data)
    
    # Step 4: Setup DataLoader for batching
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Step 5: Initialize Model
    INPUT_SIZE = WINDOW_SIZE * len(ALPHABET) # 13 * 21 = 273
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 3
    
    model = MiniFoldFFNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    
    # Step 6: Train Model
    trained_model = train_model(model, train_loader, num_epochs=10, lr=0.001)
    
    # Step 7: Run Inference Test
    test_input = "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"
    expected_output = "CCCCCCCCHHHHHHHHHHHHHHCECCCCCC"
    
    prediction = predict_protein_structure(test_input, trained_model, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    print(f"--- INFERENCE TEST ---")
    print(f"Input:    {test_input}")
    print(f"Expected: {expected_output}")
    print(f"Predicted:{prediction}")