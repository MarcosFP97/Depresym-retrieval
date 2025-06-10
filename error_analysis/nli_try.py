from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')


### SELF-DISLIKE
# features = tokenizer([('So should I dislike myself?', 'I dislike myself.'), \
#                       ('I hated myself in that moment.', 'I dislike myself.')],  padding=True, truncation=True, return_tensors="pt")

##### SADNESS
features = tokenizer([('I feel like I should be miserable, like I should be crying or upset in any way.', ' I am so sad or unhappy that I can\'t stand it'), \
                      ('I don\'t like being sad', ' I am so sad or unhappy that I can\'t stand it')],  padding=True, truncation=True, return_tensors="pt")


#### PESIMISSM
# features = tokenizer([('Currently I dont know what to do with my life.', 'I feel my future is hopeless and will only get worse.'), \
#                       ('I\'m afraid I will lose my youth because of depression.', 'I feel my future is hopeless and will only get worse.')],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    # scores[0][2] -= 100  # Restamos un valor grande para que sea casi 0 tras softmax
    print(scores)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)
probabilities = torch.nn.functional.softmax(scores, dim=-1)
print(probabilities)

if probabilities[0][0] > probabilities[1][0]:
    print("La frase irrelevante tiene más probabilidad de contradicción")
    