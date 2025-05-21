import random
import json
import faiss
from datasets import Dataset, load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from itertools import permutations

SR_MODEL = SentenceTransformer("all-mpnet-base-v2", device='cuda:0')#, device=torch.device("mps"))

def index(items: list):
    embeddings = SR_MODEL.encode(
        items,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32") 
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search_sim_symptom(
    idx:object,
    query:str,
    items:list
):
    query_embedding = SR_MODEL.encode([query], convert_to_numpy=True, device='cuda:0', normalize_embeddings=True)#.to('cuda:0')
    D, I = idx.search(query_embedding, k=2) ### we get the second most similar, to exclude similarity with itself
    return items[I[0][1]]


### LOAD BDI-QUESTIONNAIRE
bdi = {}
items = []
with open('../dataset_format_beir/queries_BDI_item.jsonl') as f:
    data = f.readlines()
    for line in data:
        dd = json.loads(line)
        items.append(dd['text'])
        splits = dd['text'].split('\n\t')
        symptom = splits[0].replace(':','')
        bdi[symptom] = []
        for i in range(1, len(splits)):
            option = splits[i].replace('\n','')
            bdi[symptom].append(option)

idx = index(items) ### create FAISS index

## BDI QUESTIONNAIRE
# train_dataset = {"anchor": [], "positive": [], "negative": []}
# symptoms = bdi.keys()
# random.seed(42)
# for k in bdi.keys():
#     options = bdi[k]
#     combinations = list(permutations(options, 2))  # Genera todas las permutaciones de 2 elementos
#     for comb in combinations:
#         train_dataset["anchor"].append(comb[0])
#         train_dataset["positive"].append(comb[1])
#         sim_symptom = search_sim_symptom(idx, k, items) #### we search by symptom title (e.g. "Sadness")
#         symptom_key = sim_symptom.split('\n\t')[0].replace(':','')
#         #random_symptom = random.choice([x for x in symptoms if x != k])
#         random_options = bdi[symptom_key]
#         random_negative = random.choice(random_options) ### from the most similar symptom, we pick a random option
#         train_dataset["negative"].append(random_negative) ### random_negatives strategy
# train_dataset = Dataset.from_dict(train_dataset)


#### SYNT DATA
train_dataset = {"anchor": [], "positive": [], "negative": []}
with open('gpt4-data.json') as f:
  SYNT_DATA = json.load(f)
symptoms = SYNT_DATA.keys()
random.seed(42)
for k in SYNT_DATA.keys():
    sents = SYNT_DATA[k]
    combinations = list(permutations(sents, 2))  # Genera todas las permutaciones de 2 elementos
    for comb in combinations:
        train_dataset["anchor"].append(comb[0])
        train_dataset["positive"].append(comb[1])
        sim_symptom = search_sim_symptom(idx, k, items) #### we search by symptom title (e.g. "Sadness")
        symptom_key = sim_symptom.split('\n\t')[0].replace(':','')
        # random_symptom = random.choice([x for x in symptoms if x != k])
        random_options = SYNT_DATA[symptom_key]
        random_negative = random.choice(random_options) ### from the most similar symptom, we pick a random option
        train_dataset["negative"].append(random_negative) ### random_negatives strategy

examples = list(zip(train_dataset["anchor"], train_dataset["positive"], train_dataset["negative"]))
# Barajar los ejemplos
random.shuffle(examples)
# Desempaquetar de nuevo en listas separadas
anchor, positive, negative = zip(*examples)

# Crear el dataset barajado
shuffled_dataset = Dataset.from_dict({
    "anchor": list(anchor),
    "positive": list(positive),
    "negative": list(negative)
})

for i in range(len(shuffled_dataset)):
    print(f'{shuffled_dataset["anchor"][i]}-{shuffled_dataset["positive"][i]}-{shuffled_dataset["negative"][i]}')

print(len(shuffled_dataset))

### MNLI ORIGIINAL EXAMPLE
# mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
# mnli = mnli.remove_columns("idx")
# mnli = mnli.filter(lambda x: True if x['label'] == 0 else False)

# # Prepare data and add a soft negative
# train_dataset = {"anchor": [], "positive": [], "negative": []}
# soft_negatives = mnli["hypothesis"]
# random.shuffle(soft_negatives)
# for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
#     train_dataset["anchor"].append(row["premise"])
#     train_dataset["positive"].append(row["hypothesis"])
#     train_dataset["negative"].append(soft_negative)
# shuffled_dataset = Dataset.from_dict(train_dataset)
# print(f'Num of training instances: {len(shuffled_dataset)}')

val_sts = load_dataset('glue', 'stsb', split='validation') # # Create an embedding similarity evaluator for stsb
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

# embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda:0') # # Define model
train_loss = losses.MultipleNegativesRankingLoss(model=SR_MODEL) # # Loss function


# # # # Define the training arguments
args = SentenceTransformerTrainingArguments( #### pensar si tiene sentido usar WandDB
    output_dir="./models/gpt4-sim-logging",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# # # Train model
trainer = SentenceTransformerTrainer(
    model=SR_MODEL,
    args=args,
    train_dataset=shuffled_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
SR_MODEL.save('./models/gpt4-sim-model')
print(evaluator(SR_MODEL))