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

def evaluate_participant(
    results_path:str,
    corpus:object,
    queries:object,
    qrels:object
):
    results = pd.read_csv(results_path, names=["index", "query", "Q0", "s_id", "rank", "score", "run"] , sep='\t')
    # error_analysis = pd.DataFrame(columns=["opción", "sentence_id", "sentence", "rel_score"])
    
    formatted_results = {}
    for query, sid, score in zip(results["query"], results["s_id"], results["score"]):
        if not str(query) in formatted_results:
            formatted_results[str(query)] = {}
        if sid in corpus:
            sent = corpus[sid]['text']
            disor_prob = disorbert_pred(sent)
            formatted_results[str(query)][sid] = 0.8*score + 0.2*disor_prob
        else:
            formatted_results[str(query)][sid] = score


    ### BUCLE QUE ITERE POR RESULTS Y HAGA UN VOTING SCORE

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
    #print(qrels['1'].keys())
    print(formatted_results['1'])
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, formatted_results, [1, 3, 5, 10, 100, 1000])
    return ndcg, _map, recall, precision
    

if __name__=="__main__":
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/queries.jsonl", "../dataset_format_beir/qrels.tsv")
    results_path = "../participants_results/formula.tsv"
    ndcg, _map, recall, precision = evaluate_participant(results_path, corpus, queries, qrels)
    print(_map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
         recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"])
