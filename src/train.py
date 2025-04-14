import random
import json
import faiss
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import losses, SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from itertools import permutations

SR_MODEL = SentenceTransformer("all-mpnet-base-v2", device='cuda')

SYNT_DATA = {
  "Sadness": [
    "I feel sad most days",
    "I don't experience sadness often",
    "I am not a sad person at all",
    "I cry easily and frequently",
    "Crying is rare for me",
    "I have been feeling down lately",
    "My mood has remained stable",
    "Sadness is a normal part of my life",
    "I don't feel much sadness",
    "I've had some recent sad moments"
  ],
  "Pessimism": [
    "I feel hopeless all the time",
    "I am not pessimistic at all",
    "Things will never get better for me",
    "I'm optimistic about my future",
    "I expect bad things to happen",
    "Good things usually happen to me",
    "Life is just too hard and unfair",
    "Things will eventually turn around",
    "I'm a pessimist at heart",
    "I try not to think negatively"
  ],
  "Suicidal Thoughts or Wishes": [
    "I often feel like I don't want to be alive anymore",
    "The thought of suicide has never crossed my mind",
    "Sometimes I wish I could just disappear",
    "I've had thoughts of ending my life, but they're fleeting",
    "I would never consider taking my own life",
    "I feel so hopeless that death might seem preferable",
    "Suicide is a tempting option for me",
    "The idea of suicide has no appeal to me",
    "Sometimes I imagine myself dead",
    "Death seems like a peaceful escape"
  ],
  "Past Failure": [
    "My past failures still haunt me",
    "I've learned from my mistakes and moved on",
    "I don't dwell on past errors",
    "The memory of my past failures is a constant reminder",
    "My past experiences have been successful",
    "Regret is a natural part of life for me",
    "I've never had any major setbacks",
    "Past failures are a normal part of learning",
    "I try not to think about my past mistakes",
    "My past accomplishments are what motivate me"
  ],
  "Loss of Pleasure": [
    "I no longer find joy in things I used to enjoy",
    "Pleasure is still an important part of my life",
    "There's nothing that brings me pleasure anymore",
    "Small pleasures still bring a smile to my face",
    "Life is too hard and uninteresting for me",
    "The idea of having fun doesn't excite me",
    "I've lost interest in most activities",
    "I try to find joy in the little things",
    "My sense of humor has been affected",
    "Some things still bring me pleasure"
  ],
  "Guilty Feelings": [
    "I often feel guilty about something",
    "Guilt is a rare emotion for me",
    "I've never felt genuinely sorry for anything",
    "Regret and guilt are normal feelings for me",
    "My conscience bothers me frequently",
    "I don't experience much guilt or remorse",
    "There's one thing in my past that I deeply regret",
    "Guilt is a natural part of human life",
    "I try not to dwell on past mistakes",
    "I rarely feel guilty about anything"
  ],
  "Punishment Feelings": [
    "I punish myself for my failures",
    "Self-punishment is never an option for me",
    "I've learned to accept my mistakes and move forward",
    "I'm too hard on myself when I make errors",
    "The thought of self-punishment doesn't appeal to me",
    "I try not to think about past failures",
    "Punishing myself is a normal part of learning",
    "Self-acceptance is my preferred approach",
    "I'm more forgiving towards myself now",
    "I don't believe in punishing oneself"
  ],
  "Self-Dislike": [
    "I often dislike and criticize myself",
    "I have a positive self-image",
    "I'm generally satisfied with who I am",
    "There's something about me that I strongly dislike",
    "I try not to dwell on negative thoughts about myself",
    "I've learned to accept my flaws and strengths",
    "Self-acceptance is difficult for me",
    "I'm more self-critical than others are of me",
    "I believe in being kind to oneself",
    "There's nothing I dislike about myself"
  ],
  "Self-Criticalness": [
    "I am very critical of my own thoughts and actions",
    "I don't tend to be overly self-critical",
    "I'm generally satisfied with the way I handle things",
    "I've never been overly critical towards myself",
    "There's something about me that I strongly criticize",
    "Self-acceptance is difficult for me",
    "I am more critical of others than myself",
    "I believe in being kind to oneself",
    "I'm too hard on myself when I make errors",
    "Self-criticism doesn't come naturally to me"
  ],
  "Crying": [
    "I cry frequently and easily",
    "Crying is rare for me",
    "There's nothing that brings tears to my eyes anymore",
    "Small things can still bring a tear to my eye",
    "I've lost interest in crying as an emotional release",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Crying is a normal part of human life",
    "I don't cry often, but I do from time to time",
    "There's still something that can bring me to tears"
  ],
  "Agitation": [
    "I feel agitated and restless most days",
    "I am generally calm and composed",
    "There's nothing that really irritates me anymore",
    "Small things can still get under my skin",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Agitation is a normal part of human life",
    "I try not to let things bother me",
    "I'm more irritable than I used to be",
    "There's nothing that agitates me"
  ],
  "Loss of Interest": [
    "I've lost interest in most activities and hobbies",
    "Interest is still an important part of my life",
    "There's nothing that really excites or interests me anymore",
    "Small things can still bring a spark to my day",
    "Life is just too hard and uninteresting for me",
    "My sense of purpose has been lost",
    "The idea of having fun doesn't excite me",
    "I try not to lose interest in things I care about",
    "There's something that still holds my attention"
  ],
  "Indecisiveness": [
    "I often find myself unable to make decisions",
    "Decisions come easily for me",
    "I've become more decisive over time",
    "There's nothing that makes decision-making difficult for me",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Indecisiveness is a normal part of human life",
    "I try not to let fear guide my decisions",
    "I'm more indecisive than I used to be",
    "There's nothing that makes decision-making difficult for me"
  ],
  "Worthlessness": [
    "I feel worthless and without value",
    "I have a sense of self-worth and worth",
    "There's something about me that I strongly dislike",
    "I've learned to accept my flaws and strengths",
    "Life is just too hard and uninteresting for me",
    "My sense of purpose has been lost",
    "The idea of having value doesn't appeal to me",
    "Worthlessness is a normal part of human life",
    "I believe in being kind to oneself",
    "There's nothing that makes me feel worthless"
  ],
  "Loss of Energy": [
    "I often feel exhausted and lacking energy",
    "Energy is still an important part of my life",
    "There's nothing that really drains my energy anymore",
    "Small things can still give me a boost of energy",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Loss of energy is a normal part of human life",
    "I try not to let fatigue get the best of me",
    "I'm more tired than I used to be",
    "There's nothing that takes away my energy"
  ],
  "Changes in Appetite": [
    "My appetite has changed significantly",
    "Appetite is still an important part of my life",
    "There's nothing that really affects my appetite anymore",
    "Small things can still affect the way I eat",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Changes in appetite are a normal part of human life",
    "I try not to let changes in appetite affect me",
    "I'm more sensitive to food now",
    "There's nothing that changes my appetite"
  ],
  "Changes in Sleeping Pattern": [
    "I often experience sleep disturbances and insomnia",
    "Sleep is still an important part of my life",
    "There's nothing that really affects the quality of my sleep anymore",
    "Small things can still affect the way I sleep",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Changes in appetite are a normal part of human life",
    "I try not to let changes in sleep affect me",
    "I'm more sensitive to my environment now",
    "There's nothing that disturbs my sleep"
  ],
  "Fatigue": [
    "I often feel fatigued and lacking energy",
    "Energy is still an important part of my life",
    "There's nothing that really drains my energy anymore",
    "Small things can still give me a boost of energy",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Fatigue is a normal part of human life",
    "I try not to let fatigue get the best of me",
    "I'm more tired than I used to be",
    "There's nothing that takes away my energy"
  ],
  "Loss of Interest in Sex": [
    "My interest in sex has decreased significantly",
    "Sex is still an important part of my life",
    "There's nothing that really affects the way I think about sex anymore",
    "Small things can still affect the way I feel about intimacy",
    "Life is just too hard and overwhelming",
    "My emotions have become numb",
    "Loss of interest in sex is a normal part of human life",
    "I try not to let changes in interest in sex affect me",
    "I'm more sensitive to my environment now",
    "There's nothing that affects the way I think about sex"
  ],
"Irritability": [
        "I get annoyed easily when things don't go my way",
        "I am not an irritable person, I can handle frustration well",
        "Sometimes I feel like yelling at people who annoy me",
        "I try to keep calm even when faced with frustrating situations",
        "It's common for me to lose my temper over little things",
        "I'm a pretty chill person and don't get irritated often",
        "When someone is being really annoying, I feel myself getting upset",
        "Most days, I can handle stress without becoming irritable",
        "Sometimes I just want to blow up at people who are driving me crazy",
        "For the most part, I'm a pretty even-tempered person"
    ],
    "Concentration Difficulty": [
        "I often find myself having trouble focusing on tasks",
        "My mind is sharp and I can concentrate easily",
        "Sometimes it takes me a while to get into a task because my focus wanders",
        "I tend to be able to stay focused on what's important",
        "It's not uncommon for me to daydream during work or school",
        "I have no trouble staying focused on things that interest me",
        "When I'm stressed, it's hard for me to concentrate",
        "For the most part, my concentration is pretty good",
        "Sometimes distractions can make it difficult for me to focus",
        "Most of the time, my mind stays sharp and focused"
    ]
}

def index(
    items:list
):
    embeddings = SR_MODEL.encode(items, convert_to_numpy=True, device='cuda', normalize_embeddings=True)#.to('cuda:0')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings)
    return gpu_index

def search_sim_symptom(
    idx:object,
    query:str,
    items:list
):
    query_embedding = SR_MODEL.encode([query], convert_to_numpy=True, device='cuda', normalize_embeddings=True)#.to('cuda:0')
    D, I = idx.search(query_embedding, k=2) ### we get the second most similar, to exclude similarity with itself
    return items[I[0][1]]


#### LOAD BDI-QUESTIONNAIRE
# bdi = {}
# items = []
# with open('../dataset_format_beir/queries_BDI_item.jsonl') as f:
#     data = f.readlines()
#     for line in data:
#         dd = json.loads(line)
#         items.append(dd['text'])
#         splits = dd['text'].split('\n\t')
#         symptom = splits[0].replace(':','')
#         bdi[symptom] = []
#         for i in range(1, len(splits)):
#             option = splits[i].replace('\n','')
#             bdi[symptom].append(option)

# idx = index(items) ### create FAISS index

# Prepare data and add a soft negative
# train_dataset = {"anchor": [], "positive": [], "negative": []}
# symptoms = bdi.keys()
# random.seed(42)
# for k in bdi.keys():
#     options = bdi[k]
#     combinations = list(permutations(options, 2))  # Genera todas las permutaciones de 2 elementos
#     for comb in combinations:
#         train_dataset["anchor"].append(comb[0])
#         train_dataset["positive"].append(comb[1])
#         # sim_symptom = search_sim_symptom(idx, k, items) #### we search by symptom title (e.g. "Sadness")
#         # symptom_key = sim_symptom.split('\n\t')[0].replace(':','')
#         random_symptom = random.choice([x for x in symptoms if x != k])
#         random_options = bdi[random_symptom]
#         random_negative = random.choice(random_options) ### from the most similar symptom, we pick a random option
#         train_dataset["negative"].append(random_negative) ### random_negatives strategy
# train_dataset = Dataset.from_dict(train_dataset)


train_dataset = {"anchor": [], "positive": [], "negative": []}
symptoms = SYNT_DATA.keys()
random.seed(42)
for k in SYNT_DATA.keys():
    sents = SYNT_DATA[k]
    combinations = list(permutations(sents, 2))  # Genera todas las permutaciones de 2 elementos
    for comb in combinations:
        train_dataset["anchor"].append(comb[0])
        train_dataset["positive"].append(comb[1])
        # sim_symptom = search_sim_symptom(idx, k, items) #### we search by symptom title (e.g. "Sadness")
        # symptom_key = sim_symptom.split('\n\t')[0].replace(':','')
        random_symptom = random.choice([x for x in symptoms if x != k])
        random_options = SYNT_DATA[random_symptom]
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

# val_sts = load_dataset('glue', 'stsb', split='validation') # # Create an embedding similarity evaluator for stsb
# evaluator = EmbeddingSimilarityEvaluator(
#     sentences1=val_sts["sentence1"],
#     sentences2=val_sts["sentence2"],
#     scores=[score/5 for score in val_sts["label"]],
#     main_similarity="cosine"
# )

# embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda:0') # # Define model
# train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model) # # Loss function


# # # # Define the training arguments
# args = SentenceTransformerTrainingArguments( #### pensar si tiene sentido usar WandDB
#     output_dir="./models/three-epochs-random-logging",
#     num_train_epochs=3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     warmup_steps=100,
#     fp16=True,
#     eval_steps=100,
#     logging_steps=100,
# )

# # # # Train model
# trainer = SentenceTransformerTrainer(
#     model=embedding_model,
#     args=args,
#     train_dataset=shuffled_dataset,
#     loss=train_loss,
#     evaluator=evaluator
# )
# trainer.train()
# embedding_model.save('./models/three-epochs-random-model')
# print(evaluator(embedding_model))