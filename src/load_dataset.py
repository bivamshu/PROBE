import pandas as pd

# ── Helper ────────────────────────────────────────────────────────────────────
def load_sheet(path, sheet_name):
    """Load a dataset sheet, skipping the 3-row title/description header."""
    df = pd.read_excel(path, sheet_name=sheet_name, header=3)
    return df.dropna(how="all").reset_index(drop=True)


# ── Load all three sheets ─────────────────────────────────────────────────────
DATA_PATH = "../data/semantic_roles_probing_dataset_v3_0_MIXED.xlsx"
en_df    = load_sheet(DATA_PATH, "English")
np_df    = load_sheet(DATA_PATH, "Nepali")
ctrl_df  = load_sheet(DATA_PATH, "Nepali_Control")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"  English sheet   : {len(en_df):>5} rows × {en_df.shape[1]} columns")
print(f"  Nepali sheet    : {len(np_df):>5} rows × {np_df.shape[1]} columns")
print(f"  Control sheet   : {len(ctrl_df):>5} rows × {ctrl_df.shape[1]} columns")
print(f"  Total sentences : {len(en_df) + len(np_df) + len(ctrl_df):>5}")


# ── Column inspection ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print("\nEnglish columns:")
for col in en_df.columns:
    print(f"  {col}")

print("\nNepali columns (extra: sentence_romanized):")
for col in np_df.columns:
    print(f"  {col}")


# ── Key distributions ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY DISTRIBUTIONS")
print("=" * 60)

for label, df in [("English", en_df), ("Nepali", np_df)]:
    print(f"\n── {label} ──")
    for col in ["tense", "word_order", "verb_category"]:
        print(f"  {col}:")
        for val, count in df[col].value_counts().items():
            print(f"    {val:<30} {count}")

print("\n── Nepali le_present (ergative marker) ──")
print(np_df["le_present"].value_counts().to_string())
print("  (Yes = past tense only; No = present & future — core experimental variable)")

print("\n── Control sheet ──")
print(f"  tense    : {ctrl_df['tense'].unique()}")
print(f"  le_present: {ctrl_df['le_present'].unique()}")
print(f"  patient_word nulls: {ctrl_df['patient_word'].isna().sum()} "
      f"(intransitive — no patient expected)")


# ── Sample sentences ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE SENTENCES")
print("=" * 60)

print("\nEnglish — one row per tense:")
for tense in ["past", "present", "future"]:
    row = en_df[en_df["tense"] == tense].iloc[0]
    print(f"  [{tense:>7}]  {row['sentence']}")
    print(f"            agent={row['agent_word']} (pos {row['agent_position']}), "
          f"patient={row['patient_word']} (pos {row['patient_position']})")

print("\nNepali — one row per tense (showing -ले status):")
for tense in ["past", "present", "future"]:
    row = np_df[np_df["tense"] == tense].iloc[0]
    print(f"  [{tense:>7}]  {row['sentence_devanagari']}")
    print(f"            {row['sentence_romanized']}")
    print(f"            agent={row['agent_word']} (pos {row['agent_position']}), "
          f"le_present={row['le_present']}")

print("\nControl (intransitive, no -ले):")
row = ctrl_df.iloc[0]
print(f"  {row['sentence_devanagari']}  /  {row['sentence_romanized']}")
print(f"  agent={row['agent_word']} (pos {row['agent_position']}), "
      f"patient=None (intransitive)")


# ── Data quality checks ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA QUALITY")
print("=" * 60)

issues = 0
for label, df in [("English", en_df), ("Nepali", np_df)]:
    nulls = df[["sentence_id","agent_word","agent_position",
                "patient_word","patient_position"]].isnull().sum()
    if nulls.sum() > 0:
        print(f"  {label}: unexpected nulls → {nulls[nulls > 0].to_dict()}")
        issues += 1

if issues == 0:
    print("  No missing values in core fields. Dataset is clean.")

# Confirm le_present matches tense in Nepali
past_le    = np_df[np_df["tense"] == "past"]["le_present"].value_counts().to_dict()
nonpast_le = np_df[np_df["tense"] != "past"]["le_present"].value_counts().to_dict()
print(f"\n  Nepali past tense   le_present values: {past_le}")
print(f"  Nepali pres/future  le_present values: {nonpast_le}")
print("  (Past should be all Yes; pres/future all No — per Nepali grammar)")

print("\n✓ Step 2 complete. Ready to proceed to tokenization.\n")