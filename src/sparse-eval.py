import argparse
import requests 
import os, pathlib, json
from tqdm import trange

from DeepCT.deepct import run_deepct 
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
from beir.generation.models import QGenModel

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


def document_expansion(
    corpus: object  
):
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    model_path = "castorini/doc2query-t5-base-msmarco"
    qgen_model = QGenModel(model_path, use_fast=False)
    gen_queries = {} 
    num_return_sequences = 3 # We have seen 3-5 questions being diverse!
    batch_size = 180 # bigger the batch-size, faster the generation!

    for start_idx in trange(0, len(corpus_list), batch_size, desc='question-generation'):            
        
        size = len(corpus_list[start_idx:start_idx + batch_size])
        ques = qgen_model.generate(
            corpus=corpus_list[start_idx:start_idx + batch_size], 
            ques_per_passage=num_return_sequences,
            max_length=64,
            top_p=0.95,
            top_k=10)
        
        for idx in range(size):
            start_id = idx * num_return_sequences
            end_id = start_id + num_return_sequences
            gen_queries[corpus_ids[start_idx + idx]] = ques[start_id: end_id]

    return gen_queries

def format_pyserini_docT5(
    gen_queries:dict
):
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join('../dataset_format_beir', pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            query_text = " ".join(gen_queries[doc_id])
            data = {"id": doc_id, "title": title, "contents": text, "queries": query_text}
            json.dump(data, fOut)
            fOut.write('\n')

    with open(os.path.join("../dataset_format_beir/", "pyserini.jsonl"), "rb") as fIn:
        requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    index_name = "beir/depresym" # beir/scifact
    requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})


def configure_deepct(
        
):
    base_model_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "models")
    bert_base_dir = util.download_and_unzip(base_model_url, out_dir) ### downloading bert-base model

    model_url = "http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/outputs/marco.zip" 
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "models")
    checkpoint_dir = util.download_and_unzip(model_url, out_dir) ### downloading DeepCT checkpoint

    if not os.path.isfile(os.path.join('../dataset_format_beir', "deepct.jsonl")):
        ################################
        #### Command-Line Arugments ####
        ################################
        run_deepct.FLAGS.task_name = "beir"                                                     # Defined a seperate BEIR task in DeepCT. Check out run_deepct.
        run_deepct.FLAGS.do_train = False                                                       # We only want to use the code for inference.
        run_deepct.FLAGS.do_eval = False                                                        # No evaluation.
        run_deepct.FLAGS.do_predict = True                                                      # True, as we would use DeepCT model for only prediction.
        run_deepct.FLAGS.data_dir = os.path.join('../dataset_format_beir', "sentences.jsonl")                     # Provide original path to corpus data, follow beir format.
        run_deepct.FLAGS.vocab_file = os.path.join(bert_base_dir, "vocab.txt")                  # Provide bert-base-uncased model vocabulary.
        run_deepct.FLAGS.bert_config_file = os.path.join(bert_base_dir, "bert_config.json")     # Provide bert-base-uncased config.json file.
        run_deepct.FLAGS.init_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-65816")     # Provide DeepCT MSMARCO model (bert-base-uncased) checkpoint file.
        run_deepct.FLAGS.max_seq_length = 350                                                   # Provide Max Sequence Length used for consideration. (Max: 512)
        run_deepct.FLAGS.train_batch_size = 60                                                 # Inference batch size, Larger more Memory but faster!
        run_deepct.FLAGS.output_dir = '../dataset_format_beir'                                                 # Output directory, this will contain two files: deepct.jsonl (output-file) and predict.tf_record
        run_deepct.FLAGS.output_file = "deepct.jsonl"                                           # Output file for storing final DeepCT produced corpus.
        run_deepct.FLAGS.m = 100                                                                # Scaling parameter for DeepCT weights: scaling parameter > 0, recommend 100
        run_deepct.FLAGS.smoothing = "sqrt"                                                     # Use sqrt to smooth weights. DeepCT Paper uses None.
        run_deepct.FLAGS.keep_all_terms = True                                                  # Do not allow DeepCT to delete terms.

        # Runs DeepCT model on the corpus.jsonl
        run_deepct.main()

def format_pyserini_deepct(
        
):
    with open(os.path.join("../dataset_format_beir/", "deepct.jsonl"), "rb") as fIn:
        requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    index_name = "beir/depresym" # beir/scifact
    requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

def evaluate_docT5(
    queries:object,
    qrels:object     
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_deepct(
    queries:object,
    qrels:object
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_sparta(
    corpus:object,
    queries:object,
    qrels:object
):
    model_path = "BeIR/sparta-msmarco-distilbert-base-v1"
    sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
    retriever = EvaluateRetrieval(sparse_model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs='?', default="queries") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/"+str(args.query)+".jsonl", "../dataset_format_beir/qrels.tsv")
    
    #### DeepCT
    # configure_deepct()
    # format_pyserini_deepct()
    # ndcg, _map, recall, precision = evaluate_deepct(queries, qrels)
    # with open("../baselines/deepct.txt",'a+') as f:
    #     print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
    #     f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))
    
    #### SPARTA
    # ndcg, _map, recall, precision = evaluate_sparta(corpus, queries, qrels)
    # with open("../baselines/sparta.txt",'a+') as f:
    #     print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
    #     f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))

    #### DocT5Query
    document_expansion(corpus)
    format_pyserini_docT5()
    ndcg, _map, recall, precision = evaluate_docT5(queries, qrels)
    with open("../baselines/docT5.txt",'a+') as f:
        print("Ndcg:", ndcg, "MAP:", _map, "Recall:", recall, "Precision:", precision)
        f.write("\nNdcg:"+ json.dumps(ndcg)+ " MAP:"+ json.dumps(_map) + " Recall:"+ json.dumps(recall) + " Precision:"+ json.dumps(precision))