import argparse
import json
from csv import writer

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

def evaluate_sentence_transf(
    model_name:str,
    corpus:object,
    queries:object,
    qrels:object
):
    model = DRES(models.SentenceBERT(model_name))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="sadness") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/options/qrels/qrels_"+str(args.symptom)+".tsv")
    sr_model = 'multi-qa-mpnet-base-dot-v1'
    ndcg, _map, recall, precision = evaluate_sentence_transf(sr_model, corpus, queries, qrels)
    row = ["multi-qa-mpnet-base-dot-v1", args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../baselines/options/output.csv",'a+') as f:
        writer_object = writer(f)
        writer_object.writerow(row)
        f.close()