from transformers import AutoTokenizer
import pandas as pd

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

# Load dataset
# Note: Adding header=3 to properly skip the title rows of your v3.0 sheets
df = pd.read_excel(
    "data/semantic_roles_probing_dataset_v3_0_MIXED.xlsx", 
    sheet_name="Nepali",  # Swap to "English" or "Nepali_Control" to test dynamism
    header=3
)
df = df.dropna(how="all").reset_index(drop=True)

# Take the first example row
row = df.iloc[0]

# ── OPTION C: DYNAMIC COLUMN DETECTION ────────────────────────────────────────
# Check which column exists in the current sheet so the script never throws a KeyError
if "sentence" in row and pd.notna(row["sentence"]):
    sentence = row["sentence"]
elif "sentence_devanagari" in row and pd.notna(row["sentence_devanagari"]):
    sentence = row["sentence_devanagari"]
elif "sentence_romanized" in row and pd.notna(row["sentence_romanized"]):
    sentence = row["sentence_romanized"]
else:
    raise KeyError("Could not find a valid text sentence column in this sheet row.")
# ──────────────────────────────────────────────────────────────────────────────

print("Sentence:")
print(sentence)

# Split into words (keeps whitespace grouping intact for word_ids mapping)
words = str(sentence).split()

print("\nWhitespace words:")
print(words)

# Tokenize while keeping pre-split word mapping
encoding = tokenizer(
    words,
    is_split_into_words=True,
    return_tensors="pt"
)

print("\nTokens:")
print(encoding.tokens())

print("\nWord IDs:")
print(encoding.word_ids())

def word_to_token_indices(encoding, word_position):
    word_ids = encoding.word_ids()

    token_indices = []

    for token_index, word_id in enumerate(word_ids):

        if word_id == word_position:
            token_indices.append(token_index)

    return token_indices 

#testing 

agent_tokens = word_to_token_indices(
    encoding,
    row["agent_position"]
)


patient_tokens = word_to_token_indices(
    encoding,
    row["patient_position"]
)


print("\nAgent token indices:")
print(agent_tokens)


print("\nPatient token indices:")
print(patient_tokens)