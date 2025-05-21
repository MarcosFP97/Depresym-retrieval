import argparse
import requests 
import os
import json
from csv import writer
from DeepCT.deepct import run_deepct 
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from beir.retrieval.search.sparse import SparseSearch
from beir.generation.models import QGenModel
from tqdm import trange
from beir import util, LoggingHandler
import os, pathlib, json

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

def search_bm25(
    queries:object
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    return retriever, results

def order_results(
    results:dict
):
    sorted_results = {}
    for qid, res in results.items():
        sorted_results[qid] = dict(sorted(res.items(), key=lambda item: item[1],reverse=True))

    return sorted_results

def semantic_search(
    queries:object,
    model_name:str
):
    model = DRES(models.SentenceBERT(model_name))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    return retriever, results

def search_sparta(
    queries:object,
):
    model_path = "BeIR/sparta-msmarco-distilbert-base-v1"
    sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
    retriever = EvaluateRetrieval(sparse_model)
    results = retriever.retrieve(corpus, queries)
    return retriever, results

def search_tasb(
    corpus:object,
    queries:object
):
    model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"))
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    return retriever, results

def rerank(
    model_name:str,
    retriever:object,
    results:dict,
    qrels:object,
    k:int
):
    cross_encoder_model = CrossEncoder(model_name,max_length=512) #'cross-encoder/ms-marco-MiniLM-L-6-v2'
    reranker = Rerank(cross_encoder_model, batch_size=16)
    rerank_results = reranker.rerank(corpus, queries, results, top_k=k)
    sorted_rerank_results = order_results(rerank_results)
    final_results = {}
    for qid, res in sorted_rerank_results.items():
        full_rank = list(results[qid].items())[k:]
        ll = list(res.items()) + full_rank
        dd = dict(ll)
        global_dd = {}
        count =1000
        for key in dd.keys():
            global_dd[key] = count
            count-=1
        final_results[qid] = global_dd
    print("Longitud", len(final_results['1']))
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, final_results, retriever.k_values)
    return ndcg, _map, recall, precision

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


def document_expansion(
    corpus: object  
):
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    model_path = "castorini/doc2query-t5-base-msmarco"
    qgen_model = QGenModel(model_path, use_fast=False)
    gen_queries = {} 
    num_return_sequences = 3 # We have seen 3-5 questions being diverse!
    batch_size = 32 # bigger the batch-size, faster the generation!

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


if __name__=="__main__":
    #### OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="agitation") ### With this param we select the kind of query: only the BDI item tite, the firs question, a symptom, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/options/qrels/qrels_"+str(args.symptom)+".tsv")
    
    # TITLE 
    # corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/queries.jsonl", "../dataset_format_beir/qrels.tsv")
    
    format_pyserini(corpus)
    # configure_deepct()
    # format_pyserini_deepct()
    retriever, results = search_bm25(queries)
    # retriever, results = semantic_search(queries,"paraphrase-multilingual-MiniLM-L12-v2")
    # gen_queries = document_expansion(corpus)
    # format_pyserini_docT5(gen_queries)
    # retriever, results = search_bm25(queries, qrels)
    # with open("result.json") as f:
    #     results = json.load(f)
    sorted_results = order_results(results)
    
    custom_reranker = 'cross-encoder/ms-marco-MiniLM-L-6-v2' #'/home/marcos.fernandez.pichel/PhD/cross-domain-symptom-detection/src/training/SimCSE/result/disorbert-wiki1m'
    ndcg, _map, recall, precision = rerank(custom_reranker, retriever, sorted_results, qrels, 100)
    row = ["bm25+cefull",100, args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@5"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../baselines/options/output.csv",'a+') as f:
       writer_object = writer(f)
       writer_object.writerow(row)
       f.close()