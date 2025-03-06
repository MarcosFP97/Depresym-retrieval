import argparse
import json
import pandas as pd
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

def order_results(
    results:dict
):
    sorted_results = {}
    for qid, res in results.items():
        sorted_results[qid] = dict(sorted(res.items(), key=lambda item: item[1],reverse=True))

    return sorted_results

def evaluate_retrieval(
    model_name:str,
    symptom:str,
    corpus:object,
    queries:object,
    qrels:object
):
    model = DRES(models.SentenceBERT(model_name))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    error_analysis = pd.DataFrame(columns=["opción", "sentence_id", "sentence", "rel_score"])
    ordered_results = order_results(results)
    for k,v in ordered_results.items():
        top_k=0
        for sid in v.keys():
            top_k+=1
            sentence = corpus[sid]['text']
            try:
                rel = qrels[str(k)][sid]
            except:
                rel=0
            error_analysis.loc[len(error_analysis)] = [k, sid, sentence, rel]
            if top_k==10:
                break
    error_analysis.to_csv(f'../error_analysis/{symptom}.csv', index=False)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="past failure") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/options/qrels/qrels_"+str(args.symptom)+".tsv")
    model_name = "all-mpnet-base-v2"
    ndcg, _map, recall, precision = evaluate_retrieval(model_name, str(args.symptom), corpus, queries, qrels)
    row = [model_name, args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    # with open("../baselines/options/output.csv",'a+') as f:
    #     writer_object = writer(f)
    #     writer_object.writerow(row)
    #     f.close()
