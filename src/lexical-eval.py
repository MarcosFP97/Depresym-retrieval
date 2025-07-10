import argparse
import requests 
import os
import json
import spacy
from csv import writer

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

docker_beir_pyserini = "http://0.0.0.0:8000"

def load_custom_data(
    corpus_path:str,
    query_path:str,
    qrels_path:str
):
    corpus, queries, qrels = GenericDataLoader(
                                    corpus_file=corpus_path, 
                                    query_file=query_path, 
                                    qrels_file=qrels_path).load_custom()
    
    return corpus, queries, qrels

def format_pyserini(
    corpus:dict
):
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join("../dataset_format_beir/", pyserini_jsonl), 'w+') as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

    with open(os.path.join("../dataset_format_beir/", "pyserini.jsonl"), "rb") as fIn:
        requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    index_name = "beir/depresym" # beir/scifact
    requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

pronouns_first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
nlp = spacy.load("en_core_web_sm")
def pos(
    sentence:str
):
    doc = nlp(sentence.lower())
    for token in doc:
        if token.text in pronouns_first_person and token.dep_ in {"nsubj", "nsubjpass"}:
            return 1000  # Sujeto en primera persona encontrado 
    return 0


def evaluate_bm25(
    queries:object,
    qrels:object
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)
    
    ### POS
    for q in results.keys():
        for sid, val in results[q].items():
            sent = corpus[sid]['text']
            pos_score = pos(sent)
            results[q][sid] = val + pos_score
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="sadness") ### With this param we select the kind of query: only the BDI item tite, the firs question, a symptom, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/2024/sentences_only_text.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/2024/options/qrels/qrels_"+str(args.symptom)+".tsv")
    format_pyserini(corpus)
    ndcg, _map, recall, precision = evaluate_bm25(queries, qrels)
    row = ["bm25_2024", args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../custom_sols/baselines_pos.csv",'a+') as f:
        writer_object = writer(f)
        writer_object.writerow(row)
        f.close()