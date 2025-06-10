import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "citiusLTL/DisorBERT"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda:0")  # GPU específica

# Listas de pronombres de interés
pronouns_first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
pronouns_second_person = {"you", "your", "yours"}

def disorbert_person_score(sentence, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to("cuda:0")

    # Encontrar los pronombres
    pronoun_indices = [i for i, token in enumerate(tokens) if token in pronouns_first_person or token in pronouns_second_person]
    
    if not pronoun_indices:
        return 0  # No hay pronombres, no aplicamos filtro, nos quedamos con el score de SBERT

    # Enmascarar los pronombres
    masked_input_ids = input_ids.clone()
    for idx in pronoun_indices:
        masked_input_ids[0, idx + 1] = tokenizer.mask_token_id  # +1 por tokens especiales

    # Obtener predicciones del modelo
    with torch.no_grad():
        outputs = model(masked_input_ids)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    first_person_prob = 0
    second_person_prob = 0

    for idx in pronoun_indices:
        first_probs = [probs[0, idx + 1, tokenizer.convert_tokens_to_ids(p)].item() for p in pronouns_first_person]
        second_probs = [probs[0, idx + 1, tokenizer.convert_tokens_to_ids(p)].item() for p in pronouns_second_person]

        first_person_prob += sum(first_probs)
        second_person_prob += sum(second_probs)

    # Normalización por cantidad de pronombres
    first_person_prob /= len(pronoun_indices)
    second_person_prob /= len(pronoun_indices)
    
    confidence = first_person_prob - second_person_prob # Diferencia: positivo si favorece la primera persona, negativo si favorece la segunda
    if confidence < 0: #### si ya DisorBERT dice que es una segunda persona lo más probable, se penaliza
        return confidence
    
    else: ### pero si dice que es una primera persona, tenemos que ver si realmente lo es y si no lo es la penalizamos
        real_tokens_preds = []
        for idx in pronoun_indices:
            real_token_id = input_ids[0, idx + 1]  # Token original
            real_token_prob = probs[0, idx + 1, real_token_id].item()  # Probabilidad del token original
            real_tokens_preds.append(real_token_prob)
        return sum(real_tokens_preds) / len(real_tokens_preds) 

# Ejemplo de oraciones
sentence1 = "I think I am depressed."  
sentence2 = "Do you feel guilty?" #"You feel guilty, too."  

score1 = disorbert_person_score(sentence1, model, tokenizer)
score2 = disorbert_person_score(sentence2, model, tokenizer)

print(f"Score DisorBERT (1ª persona): {score1}")
print(f"Score DisorBERT (2ª persona): {score2}")
