import argparse
import requests 
import os
import json

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
    
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs='?', default="queries") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/"+str(args.query)+".jsonl", "../dataset_format_beir/qrels.tsv")
    format_pyserini(corpus)
    ndcg, _map, recall, precision = evaluate_bm25(queries, qrels)
    with open("../baselines/anserini_bm25.txt",'a+') as f:
        print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
        f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))