import pandas as pd

file_name = 'RS126.data.txt' # <-- CHANGE THIS to your file's name!
window_size = 13
pad_length = window_size // 2

X_data = [] # The 13-letter windows
Y_data = [] # The 1-letter target shapes

print("Reading biological data...")

# 1. Open and read the file
with open(file_name, 'r') as file:
    lines = file.readlines() 

# 2. Loop through the lines 2 at a time (Sequence, then Structure)
# Let's process the first 10 proteins (20 lines) 
for i in range(0, 20, 2): 
    
    seq = lines[i].strip()       
    struct = lines[i+1].strip()  
    
    # 3. Pad the sequence with 'X'
    padded_seq = ("X" * pad_length) + seq + ("X" * pad_length)
    
    # 4. Slide the window!
    for j in range(len(seq)):
        window = padded_seq[j : j + window_size]
        target = struct[j]
        
        X_data.append(window)
        Y_data.append(target)

print(f"Successfully chopped data into {len(X_data)} training examples!\n")

# 5. Convert it into a Pandas DataFrame
df = pd.DataFrame({
    'Window_Input': X_data,
    'Target_Structure': Y_data
})

# 6. Print the first 15 rows to the VS Code Terminal
print("--- First 15 Windows ---")
print(df.head(15))