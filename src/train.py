import random
import json
import faiss
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers import SentenceTransformer, InputExample
from itertools import permutations
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from torch import nn
from torch.utils.data import DataLoader, Sampler, BatchSampler

'''
CLASES CUSTOM: PARA CREAR BATCHES DE HARD NEGATIVES Y UNA FUNCIÓN DE PÉRDIDA HÍBRIDA
'''
class CustomBatchSampler(Sampler):
    def __init__(self, dataset, cluster_similar):
        self.dataset = dataset
        self.cluster_similar = cluster_similar

        self.indices_by_cluster = {} ### índices agrupados por síntoma
        for i in range(len(dataset)):
            cluster = dataset[i]["symptom"]
            self.indices_by_cluster.setdefault(cluster, []).append(i)
        self.clusters = list(self.indices_by_cluster.keys())

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # todos los pares
        #### barajar los índices
        for orig in range(0, len(indices)):
            batch = []
            cluster = self.dataset[orig]["symptom"]
            similar_clusters = self.cluster_similar.get(cluster)
            # Sacar un par anchor-positive del cluster original
            indices_orig = self.indices_by_cluster.get(cluster, [])
            orig,_ = random.sample(indices_orig, 2)
            batch.append(orig)

            # Sacar un par anchor-positive del cluster similar
            for sim_cluster in similar_clusters:
                indices_sim = self.indices_by_cluster.get(sim_cluster, [])
                sim,_ = random.sample(indices_sim, 2)
                batch.append(sim)
            print(batch)
            yield batch


class DataLoaderWithBatchSize(torch.utils.data.DataLoader):
    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
    
    @property
    def batch_size(self):
        return self._batch_size

class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
    def get_batch_sampler(self, dataset: Dataset, batch_size: int, drop_last: bool, valid_label_columns: list[str] | None = None, generator: torch.Generator | None = None) -> BatchSampler | None:
        # Lógica personalizada para el muestreo por lotes
        return BatchSampler(SubsetRandomSampler(range(len(dataset))), batch_size=batch_size, drop_last=drop_last)

class HybridLoss(nn.Module): #### de momento no se usa
    def __init__(self, mnr_loss, triplet_loss, alpha=0.5):
        super().__init__()
        self.mnr_loss = mnr_loss
        self.triplet_loss = triplet_loss
        self.alpha = alpha
        
    def forward(self, sentence_features, labels):
        loss1 = self.mnr_loss(sentence_features, labels)
        loss2 = self.triplet_loss(sentence_features, labels)
        return self.alpha * loss1 + (1 - self.alpha) * loss2

'''
FUNCIONES AUXILIARES
'''
def index(items: list):
    search_model = SentenceTransformer("all-mpnet-base-v2", device='cuda:0')
    embeddings = search_model.encode(
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
    search_model = SentenceTransformer("all-mpnet-base-v2", device='cuda:0') ### modelo para similaridad
    query_embedding = search_model.encode([query], convert_to_numpy=True, device='cuda:0', normalize_embeddings=True)#.to('cuda:0')
    D, I = idx.search(query_embedding, k=5) ### we get the second most similar, to exclude similarity with itself
    similar_clusters = [items[i] for i in I[0][1:]]
    return similar_clusters

def my_collate(batch):
    anchors = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    return anchors, positives

'''
CARGA DE DATOS
'''
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
train_dataset = {"anchor": [], "positive": [], "symptom":[]}
symptoms = bdi.keys()
random.seed(42)
for k in bdi.keys():
    options = bdi[k]
    combinations = list(permutations(options, 2))  # Genera todas las permutaciones de 2 elementos
    for comb in combinations:
        train_dataset["anchor"].append(comb[0])
        train_dataset["positive"].append(comb[1])
        train_dataset["symptom"].append(k)
train_dataset = Dataset.from_dict(train_dataset)

#### SYNT DATA
# train_dataset = {"anchor": [], "positive": [], "negative": []}
# with open('gpt4-data.json') as f:
#   SYNT_DATA = json.load(f)
# symptoms = SYNT_DATA.keys()
# random.seed(42)
# for k in SYNT_DATA.keys():
#     sents = SYNT_DATA[k]
#     combinations = list(permutations(sents, 2))  # Genera todas las permutaciones de 2 elementos
#     for comb in combinations:
#         train_dataset["anchor"].append(comb[0])
#         train_dataset["positive"].append(comb[1])
#         sim_symptom = search_sim_symptom(idx, k, items) #### we search by symptom title (e.g. "Sadness")
#         symptom_key = sim_symptom.split('\n\t')[0].replace(':','')
#         # random_symptom = random.choice([x for x in symptoms if x != k])
#         random_options = SYNT_DATA[symptom_key]
#         random_negative = random.choice(random_options) ### from the most similar symptom, we pick a random option
#         train_dataset["negative"].append(random_negative) ### random_negatives strategy

symptom_to_similar = {}
for i, name in enumerate(symptoms):
    # Calcular similitud coseno con el resto
    sims = search_sim_symptom(idx,name, items)
    symptom_to_similar[name] = []
    for sim in sims:
        symptom_to_similar[name].append(sim.split('\n\t')[0].replace(':',''))

# print(symptom_to_similar)
sampler = CustomBatchSampler(train_dataset, symptom_to_similar)

train_dataset = [InputExample(texts=[item['anchor'], item['positive']]) for item in train_dataset]

model = SentenceTransformer("all-mpnet-base-v2")

dataloader = DataLoaderWithBatchSize(
    train_dataset,
    batch_sampler=sampler,
    collate_fn=model.smart_batching_collate,
    batch_size=6,
    shuffle = False
)


#### For debugging purposes...
# for i, batch in enumerate(dataloader):
#     print(f"Batch {i}: {batch}")
#     if i >= 3:  # Muestra solo los 4 primeros
#         break

loss = MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(dataloader, loss)],
    epochs=5,
    warmup_steps=100,
)