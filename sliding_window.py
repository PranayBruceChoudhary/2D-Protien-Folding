protein_sequence = "MVLSPADKTN"
protein_structure = "CCHHHHHCCC" # C=Coil, H=Helix

window_size = 13 

# Calculate how much padding we need on each side
# If window is 13, integer division (//) gives us 6 on each side.
pad_length = window_size // 2 

# 1. Pad the sequence with 'X'
padded_sequence = ("X" * pad_length) + protein_sequence + ("X" * pad_length)

print(f"Original: {protein_sequence}")
print(f"Padded:   {padded_sequence}\n")
print("--- Generating Sliding Windows ---\n")

# 2. Slide the window across the original protein length
# We create lists to store our data for Machine Learning later
X_data = [] # This will hold the windows (Inputs)
Y_data = [] # This will hold the shapes (Target Outputs)

for i in range(len(protein_sequence)):
    # Slice the padded sequence to get exactly 13 characters
    window = padded_sequence[i : i + window_size]
    
    # Get the target shape for the middle character
    target_shape = protein_structure[i]
    
    # Save them to our lists
    X_data.append(window)
    Y_data.append(target_shape)
    
    # Print it out so we can visualize what is happening!
    print(f"Window Input: [{window}]  -->  Target Shape to Predict: {target_shape}")
