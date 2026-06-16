from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# ==============================
# 1. Load tokenizer + model
# ==============================
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
model.eval()

# ==============================
# 2. Load dataset (Updated for Cross-Linguistic Parallel Ingestion)
# ==============================
target_sheets = ["English", "Nepali"]
dataframes = []

for sheet in target_sheets:
    sheet_df = pd.read_excel(
        "data/semantic_roles_probing_dataset_v3_0_MIXED.xlsx",
        sheet_name=sheet,
        header=3
    )
    # Filter out empty spacer padding lines using sentence_id as a pivot anchor
    sheet_df = sheet_df.dropna(subset=["sentence_id"]).reset_index(drop=True)
    dataframes.append(sheet_df)

# Combine both sheets into one unified Master DataFrame of 1,800 sentences
df = pd.concat(dataframes, ignore_index=True)

print("Total Number of Sentences Loaded Across Both Languages:")
print(len(df))

# ==============================
# 3. Helper functions
# ==============================
def word_to_token_indices(encoding, word_position):
    word_ids = encoding.word_ids()
    indices = []
    for token_index, word_id in enumerate(word_ids):
        if word_id == word_position:
            indices.append(token_index)
    return indices

def get_word_embedding(hidden_layer, token_indices):
    token_vectors = hidden_layer[0, token_indices, :]
    return torch.mean(token_vectors, dim=0)

# ==============================
# 4. Storage & Presentation Logs
# ==============================
X = []
y = []
metadata = []
presentation_logs = []  # Tracks human-readable tokens for tomorrow's demo

# ==============================
# 5. Process every sentence
# ==============================
for index, row in tqdm(df.iterrows(), total=len(df)):

    # Choose sentence column dynamically to avoid KeyErrors across languages
    if "sentence" in row and pd.notna(row["sentence"]):
        sentence = row["sentence"]
    elif "sentence_devanagari" in row and pd.notna(row["sentence_devanagari"]):
        sentence = row["sentence_devanagari"]
    elif "sentence_romanized" in row and pd.notna(row["sentence_romanized"]):
        sentence = row["sentence_romanized"]
    else:
        continue

    words = str(sentence).split()

    # Tokenize
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt"
    )

    # Get agent/patient token locations
    agent_tokens = word_to_token_indices(encoding, int(row["agent_position"]))
    patient_tokens = word_to_token_indices(encoding, int(row["patient_position"]))

    # Skip bad rows
    if len(agent_tokens) == 0 or len(patient_tokens) == 0:
        continue

    # Capture literal readable strings for presentation
    literal_tokens = encoding.tokens()
    
    # Isolate tokens matching our target positions
    agent_subwords = [literal_tokens[i] for i in agent_tokens]
    patient_subwords = [literal_tokens[i] for i in patient_tokens]

    # Format data log block (Shows clear contrast between English/Nepali token streams)
    log_entry = (
        f"Row ID: {row['sentence_id']}\n"
        f"Input Sentence:   {sentence}\n"
        f"mBERT WordPieces: {literal_tokens}\n"
        f"Word ID Mapping:  {encoding.word_ids()}\n"
        f"Target Agent:     {agent_subwords} (Token indices: {agent_tokens})\n"
        f"Target Patient:   {patient_subwords} (Token indices: {patient_tokens})\n"
        f"{'-'*80}\n"
    )
    presentation_logs.append(log_entry)

    # Print a quick preview when switching sheets to visually confirm alignment works
    if len(presentation_logs) == 1 or len(presentation_logs) == 901:
        print("\n=== LAYER EXTRACTION WORKFLOW PREVIEW ===")
        print(log_entry)

    # mBERT forward pass
    with torch.no_grad():
        outputs = model(**encoding)

    hidden = outputs.hidden_states[12]

    # Embeddings
    agent_vector = get_word_embedding(hidden, agent_tokens)
    patient_vector = get_word_embedding(hidden, patient_tokens)

    # Store Agent
    X.append(agent_vector.numpy())
    y.append(1)
    metadata.append({"sentence_id": row["sentence_id"], "role": "Agent"})

    # Store Patient
    X.append(patient_vector.numpy())
    y.append(0)
    metadata.append({"sentence_id": row["sentence_id"], "role": "Patient"})

# ==============================
# 6. Save Presentation Text Artifact
# ==============================
with open("data/tokenization_results_presentation.txt", "w", encoding="utf-8") as f:
    f.writelines(presentation_logs)

# ==============================
# 7. Convert to tensors (Optimized)
# ==============================
X = torch.from_numpy(np.stack(X))
y = torch.tensor(y)

print("\nFinished!")
print(f"All {len(presentation_logs)} sentences compiled.")
print(f"Human-readable token log saved to: 'data/tokenization_results_presentation.txt'")
print("X shape:", X.shape)
print("y shape:", y.shape)