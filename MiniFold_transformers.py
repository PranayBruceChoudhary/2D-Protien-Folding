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
        
    sequences, structures = [], []
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


def prepare_transformer_tensors(X_data, Y_data, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """
    Converts 13-character string windows into 2D integer tensors [Total_Rows, 13] for Self-Attention.
    Converts Y target letters (C, E, H) into integer labels (0, 1, 2).
    """
    print("Formatting tensors for Transformer (Integer Tokens)...")
    char_to_index = {char: idx for idx, char in enumerate(alphabet)}
    
    # 1. Store raw integer tokens directly: Shape [Total_Rows, 13]
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
# 2. TRANSFORMER MODEL ARCHITECTURE
# =========================================================

class MiniFoldTransformer(nn.Module):
    """
    Transformer-based secondary structure predictor.
    Uses Token Embeddings + Positional Embeddings + Multi-Head Self-Attention
    to predict the secondary structure of the middle amino acid.
    """
    def __init__(self, vocab_size=21, window_size=13, d_model=64, nhead=4, num_layers=2, output_size=3, dropout=0.3):
        super(MiniFoldTransformer, self).__init__()
        
        self.window_size = window_size
        self.d_model = d_model
        
        # 1. Biological Token Embedding Layer: [Batch, 13] -> [Batch, 13, d_model]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        # 2. Learned Positional Embeddings: Injects position order (0 through 12)
        self.pos_embedding = nn.Embedding(num_embeddings=window_size, embedding_dim=d_model)
        
        # 3. Transformer Encoder Block (Self-Attention + Feed Forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 128 internal hidden neurons
            dropout=dropout,
            activation='relu',
            batch_first=True  # MANDATORY: Ensures shape is [Batch, Length, Features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Classification Head (Evaluates the middle token -> C, E, or H)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape entering: [Batch, 13] (integers)
        batch_size, seq_len = x.shape
        
        # 1. Biological embeddings: [Batch, 13, 64]
        token_embeddings = self.embedding(x)
        
        # 2. Positional embeddings: [Batch, 13, 64]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeddings = self.pos_embedding(positions)
        
        # 3. Combine and apply dropout: [Batch, 13, 64]
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        
        # 4. Multi-Head Self-Attention: [Batch, 13, 64]
        x = self.transformer_encoder(x)
        
        # 5. Extract ONLY the middle amino acid (Index 6): [Batch, 64]
        middle_idx = self.window_size // 2  # 13 // 2 = 6
        middle_token = x[:, middle_idx, :]
        
        # 6. Final Classification: [Batch, 3] (Logits for C, E, H)
        out = self.fc(middle_token)
        return out


# =========================================================
# 3. TRAINING FUNCTION
# =========================================================

def train_transformer_model(model, train_loader, num_epochs=100, lr=0.0005):
    """Executes the training loop with Backpropagation, Loss tracking, and Weight updates."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- STARTING TRANSFORMER TRAINING ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            # 1. Forward Pass (Self-Attention computed in parallel!)
            outputs = model(batch_X)
            
            # 2. Loss Calculation
            loss = criterion(outputs, batch_Y)
            
            # 3. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Average Loss: {avg_loss:.4f}")
            
    print("--- TRAINING COMPLETE ---\n")
    return model


# =========================================================
# 4. INFERENCE FUNCTION
# =========================================================

def predict_protein_structure_transformer(protein_seq, trained_model, window_size=13, alphabet="ACDEFGHIKLMNPQRSTVWYX"):
    """Takes a raw protein sequence string and predicts its secondary structure string."""
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
                
    # 2. Run Forward Pass with Self-Attention
    with torch.no_grad():
        outputs = trained_model(X_test_ints) # Shape: [Length, 3]
        predicted_indices = torch.argmax(outputs, dim=1)
        
    # 3. Translate integer indices back to letters
    int_to_shape = {0: 'C', 1: 'E', 2: 'H'}
    predicted_chars = [int_to_shape[int(idx.item())] for idx in predicted_indices]
    
    return "".join(predicted_chars)


# =========================================================
# 5. MAIN SCRIPT EXECUTION
# =========================================================

if __name__ == '__main__':
    # Configuration
    FILE_PATH = 'C:/Users/mannu/2D Protien Folding/RS126.data.txt' # <-- UPDATE WITH YOUR FILE PATH
    WINDOW_SIZE = 13
    ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    OUTPUT_SIZE = 3
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.0005
    
    # Step 1: Load and parse biological data
    sequences, structures = load_and_parse_data(FILE_PATH)
    
    # Step 2: Slice Sliding Windows
    X_data, Y_data = create_sliding_windows(sequences, structures, window_size=WINDOW_SIZE, limit=50)
    
    # Step 3: Prepare Integer Tensors for Transformer
    X_ints, Y_tensor = prepare_transformer_tensors(X_data, Y_data, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    # Step 4: Setup DataLoader for Mini-Batching
    dataset = TensorDataset(X_ints, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Step 5: Initialize MiniFold Transformer Model
    model = MiniFoldTransformer(
        vocab_size=len(ALPHABET),
        window_size=WINDOW_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=0.3
    )
    
    print("Transformer model successfully initialized! Architecture summary:")
    print(model)
    
    # Step 6: Train the Transformer
    trained_model = train_transformer_model(model, train_loader, num_epochs=EPOCHS, lr=LR)
    
    # Step 7: Run Inference Test
    test_input      = "SIPPEVKFNKPFVFLMIEQNTKSPLFMGKVVNPTQK"
    expected_output = "CCCCEEECCCCEEEEEEECCCCCEEEEEEECCCCCC"
    
    prediction = predict_protein_structure_transformer(test_input, trained_model, window_size=WINDOW_SIZE, alphabet=ALPHABET)
    
    # Calculate accuracy
    matches = sum(1 for p, e in zip(prediction, expected_output) if p == e)
    accuracy = (matches / len(expected_output)) * 100
    
    print("--- TRANSFORMER INFERENCE TEST RESULTS ---")
    print(f"Input Sequence:   {test_input}")
    print(f"Expected Ground:  {expected_output}")
    print(f"Transformer Pred: {prediction}")
    print(f"Accuracy:         {accuracy:.1f}% ({matches}/{len(expected_output)} correct amino acids)")