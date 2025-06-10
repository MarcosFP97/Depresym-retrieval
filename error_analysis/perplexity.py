from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import math
import torch.nn.functional as F

model_name = "citiusLTL/DisorBERT"  # O tu modelo adaptado
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

def compute_perplexity(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    loss = 0.0
    count = 0

    for i in range(1, len(input_ids[0]) - 1):  # Saltamos [CLS] y [SEP]
        # Crear copia con máscara en la posición i
        masked_input = input_ids.clone()
        masked_input[0][i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits

        probs = F.softmax(logits[0, i], dim=-1)
        target_id = input_ids[0][i]
        token_prob = probs[target_id]

        loss += -torch.log(token_prob + 1e-10)  # Añadir epsilon para evitar log(0)
        count += 1
    ppl = torch.exp(loss / count).item()
    score = 1 / math.log(ppl + 1)
    return score

sentence = "So should I hate myself?"
ppl = compute_perplexity(sentence)
print(f"Perplexity: {ppl:.4f}")