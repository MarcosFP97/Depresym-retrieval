from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import pickle

symptoms = {
  1: "Sadness",
  2: "Pessimism",
  3: "Past Failure",
  4: "Loss of Pleasure",
  5: "Guilty Feelings",
  6: "Punishment Feelings",
  7: "Self-Dislike",
  8: "Self-Criticalness",
  9: "Suicidal Thoughts or Wishes",
  10: "Crying",
  11: "Agitation",
  12: "Loss of Interest",
  13: "Indecisiveness",
  14: "Worthlessness",
  15: "Loss of Energy",
  16: "Changes in Sleeping Pattern",
  17: "Irritability",
  18: "Changes in Appetite",
  19: "Concentration Difficulty",
  20: "Tiredness or Fatigue",
  21: "Loss of Interest in Sex"
}

qrels= pd.read_csv('../../dataset_format_beir/qrels.tsv', names=["query", "doc_id", "label"], sep='\t')
print(qrels)

corpus = {}
with open('../../dataset_format_beir/sentences.jsonl') as f:
    for line in f:
       entry = json.loads(line)
       corpus[entry["_id"]] = entry["text"]  

# print(corpus)

for k, v in symptoms.items():
    symptom_qrels = qrels[(qrels["query"]==k) & (qrels["label"]==1)]
    print(len(symptom_qrels))
    sents = []
    for doc_id in symptom_qrels["doc_id"].values:
        sents.append(corpus[doc_id])
    # print(sents[:5])
    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = sbert_model.encode(sents, normalize_embeddings=True)
    labels = [v]*len(embeddings)
    with open(f"./qrels/{v}_pre_embeddings.pkl", "wb") as f:
        pickle.dump((sents, embeddings, labels), f)
    sbert_model = SentenceTransformer('../models/contr-bdi-sim-model')
    embeddings = sbert_model.encode(sents, normalize_embeddings=True)
    with open(f"./qrels/{v}_post_embeddings.pkl", "wb") as f:
        pickle.dump((sents, embeddings, labels), f)