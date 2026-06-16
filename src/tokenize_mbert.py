from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

sentence = "रामले द्रौपदीलाई खेद्यो"

encoding = tokenizer(
    sentence, 
    return_tensors = "pt"
)

print(encoding.tokens())