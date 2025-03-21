import argparse
import json
import pandas as pd
from csv import writer
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

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

disor_tokenizer = AutoTokenizer.from_pretrained("citiusLTL/DisorBERT")
disor_model = AutoModelForMaskedLM.from_pretrained("citiusLTL/DisorBERT").to('cuda:0')
pronouns_first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
pronouns_second_person = {"you", "your", "yours"}

def disorbert_pred(
    sentence:str
):
    tokens = disor_tokenizer.tokenize(sentence)
    input_ids = disor_tokenizer.encode(sentence, max_length=512, truncation=True, return_tensors="pt").to("cuda:0")

    # Encontrar los pronombres
    pronoun_indices = [i for i, token in enumerate(tokens) if token in pronouns_first_person or token in pronouns_second_person]
    
    if not pronoun_indices:
        return 0  # No hay pronombres, no aplicamos filtro, nos quedamos con el score de SBERT

    # Enmascarar los pronombres
    masked_input_ids = input_ids.clone()
    for idx in pronoun_indices:
        masked_input_ids[0, idx + 1] = disor_tokenizer.mask_token_id  # +1 por tokens especiales

    # Obtener predicciones del modelo
    with torch.no_grad():
        outputs = disor_model(masked_input_ids)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    first_person_prob = 0
    second_person_prob = 0

    for idx in pronoun_indices:
        first_probs = [probs[0, idx + 1, disor_tokenizer.convert_tokens_to_ids(p)].item() for p in pronouns_first_person]
        second_probs = [probs[0, idx + 1, disor_tokenizer.convert_tokens_to_ids(p)].item() for p in pronouns_second_person]

        first_person_prob += sum(first_probs)
        second_person_prob += sum(second_probs)

    # Normalización por cantidad de pronombres
    first_person_prob /= len(pronoun_indices)
    second_person_prob /= len(pronoun_indices)
    
    confidence = first_person_prob - second_person_prob # Diferencia: positivo si favorece la primera persona, negativo si favorece la segunda
    if confidence < 0: #### si ya DisorBERT dice que es una segunda persona lo más probable, se penaliza
        return confidence
    
    else: ### pero si dice que es una primera persona, tenemos que ver si realmente lo es y si no lo es la penalizamos
        real_tokens_preds = []
        for idx in pronoun_indices:
            real_token_id = input_ids[0, idx + 1]  # Token original
            real_token_prob = probs[0, idx + 1, real_token_id].item()  # Probabilidad del token original
            real_tokens_preds.append(real_token_prob)
        return sum(real_tokens_preds) / len(real_tokens_preds) 

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

    ### BUCLE QUE ITERE POR RESULTS Y HAGA UN VOTING SCORE
    for q in results.keys():
        for sid, val in results[q].items():
            sent = corpus[sid]['text']
            print("SENT", sent)
            disor_prob = disorbert_pred(sent)
            results[q][sid] = 0.8*val + 0.2*disor_prob

    # ordered_results = order_results(results)
    # for k,v in ordered_results.items():
    #     top_k=0
    #     for sid in v.keys():
    #         top_k+=1
    #         sentence = corpus[sid]['text']
    #         try:
    #             rel = qrels[str(k)][sid]
    #         except:
    #             rel=0
    #         error_analysis.loc[len(error_analysis)] = [k, sid, sentence, rel]
    #         if top_k==10:
    #             break
    # error_analysis.to_csv(f'../error_analysis/{symptom}_balanced_v2.csv', index=False)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="punishment feelings") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/options/qrels/qrels_"+str(args.symptom)+".tsv")
    model_name = "all-mpnet-base-v2"
    ndcg, _map, recall, precision = evaluate_retrieval(model_name, str(args.symptom), corpus, queries, qrels)
    row = ["all-mpnet-base-v2+disorbert_pred", args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../custom_sols/output.csv",'a+') as f:
        writer_object = writer(f)
        writer_object.writerow(row)
        f.close()
