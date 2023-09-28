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
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/queries.jsonl", "../dataset_format_beir/qrels.tsv")
    ndcg, _map, recall, precision = evaluate_sentence_transf("msmarco-distilbert-base-v3", corpus, queries, qrels)
    print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)