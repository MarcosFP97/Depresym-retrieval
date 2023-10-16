import argparse
import json

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

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

def evaluate_dpr(
    corpus:object,
    queries:object,
    qrels:object
):
    model = DRES(models.SentenceBERT((
        "facebook-dpr-question_encoder-multiset-base",
        "facebook-dpr-ctx_encoder-multiset-base"), batch_size=128))    
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

#### Here include more methods for ANCE, etc.

def evaluate_ance(
    corpus:object,
    queries:object,
    qrels:object
):
    model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_tasb(
    corpus:object,
    queries:object,
    qrels:object
):
    model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="sadness") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/qrels_"+str(args.symptom)+".tsv")
    
    #### DPR eval
    ndcg, _map, recall, precision = evaluate_dpr(corpus, queries, qrels)
    with open("../baselines/options/dpr.txt",'a+') as f:
        print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
        f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))
    
    #### ANCE eval
    ndcg, _map, recall, precision = evaluate_ance(corpus, queries, qrels)
    with open("../baselines/dpr.txt",'a+') as f:
        print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
        f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))

    #### TASB eval
    ndcg, _map, recall, precision = evaluate_tasb(corpus, queries, qrels)
    with open("../baselines/tasb.txt",'a+') as f:
        print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
        f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))
