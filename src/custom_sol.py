import argparse
import spacy
import json
from csv import writer
import torch
import pickle
import math
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
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

# disor_tokenizer = AutoTokenizer.from_pretrained("citiusLTL/DisorBERT")
# disor_model = AutoModelForMaskedLM.from_pretrained("citiusLTL/DisorBERT").to('cuda:0')
pronouns_first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
# pronouns_second_person = {"you", "your", "yours"}


nlp = spacy.load("en_core_web_sm")
def pos(
    sentence:str
):
    doc = nlp(sentence.lower())
    for token in doc:
        if token.text in pronouns_first_person and token.dep_ in {"nsubj", "nsubjpass"}:
            return 1000  # Sujeto en primera persona encontrado 
    return 0

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

# def fuse_scores(
#     results:dict, 
#     symptom:str
# ):
#     with open('../dataset_format_beir/queries.jsonl') as f:
#         data=f.readlines(f)
#         for ll in data:
#             dd = json.loads(ll)
#             if dd['text'].lower()

def missing_rels(
    results:dict,
    symptom:str,
    k:int=100
):
    qrels = pd.read_csv(f'../dataset_format_beir/options/qrels/qrels_{symptom}.tsv', sep='\t', names=["q_id", "doc_id", "rel_value"])
    s_irrels = qrels[(qrels["q_id"]==4) & (qrels["rel_value"]==0)]['doc_id'].values
    s_rels = qrels[(qrels["q_id"]==4) & (qrels["rel_value"]==1)]['doc_id'].values
    found_irrels = [x for x in s_irrels if x in list(results['4'].keys())[:k]] ### de momento miramos los que no coge una opción (la más extrema)
    missed_rels = [x for x in s_rels if x in list(results['4'].keys())[k:k+50]] ### de momento miramos los que no coge una opción (la más extrema)
    return found_irrels, missed_rels


# nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base').to("cuda:0")
# nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')

def nli(
    ordered_results: dict,
    queries: dict
):
    for k,v in ordered_results.items():
        top_k=0
        q_id = k
        query = queries[q_id]

        for sid, score in v.items():
            if top_k>=100 and top_k<150:
                sentence = corpus[sid]['text']
                features = nli_tokenizer([(sentence, query)],  padding=True, truncation=True, return_tensors="pt").to("cuda:0")
                nli_model.eval()
                with torch.no_grad():
                    scores = nli_model(**features).logits
                probabilities = torch.nn.functional.softmax(scores, dim=-1)
                cont_prob = probabilities[0][0]
                ent_prob = probabilities[0][1]
                if ent_prob > cont_prob:
                    ordered_results[k][sid] = 0.8*score + 0.2*ent_prob.item()
            top_k+=1

    return ordered_results

# disor_model_name = "citiusLTL/DisorBERT"  
# tokenizer = AutoTokenizer.from_pretrained(disor_model_name)
# disor_model = AutoModelForMaskedLM.from_pretrained(disor_model_name).to('cuda:0')

def compute_perplexity(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors='pt').to('cuda:0')
    loss = 0.0
    count = 0
    disor_model.eval()
    for i in range(1, len(input_ids[0]) - 1):  # Saltamos [CLS] y [SEP]
        # Crear copia con máscara en la posición i
        masked_input = input_ids.clone()
        masked_input[0][i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = disor_model(masked_input)
            logits = outputs.logits

        probs = F.softmax(logits[0, i], dim=-1)
        target_id = input_ids[0][i]
        token_prob = probs[target_id]

        loss += -torch.log(token_prob + 1e-10)  # Añadir epsilon para evitar log(0)
        count += 1
    ppl = torch.exp(loss / count).item()
    score = 1 / math.log(ppl + 1)
    return score

def evaluate_retrieval(
    model_name:str,
    corpus:object,
    queries:object,
    qrels:object
):
    model = models.SentenceBERT(model_name)
    # model.q_model = model_name  # Tu SentenceTransformer con pooling
    # model.doc_model = model_name
    model = DRES(model)
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)

    ### BUCLE QUE ITERE POR RESULTS Y HAGA UN VOTING SCORE
    for q in results.keys():
        for sid, val in results[q].items():
            sent = corpus[sid]['text']
            pos_score = pos(sent)
            results[q][sid] = val + pos_score
    
    ordered_results = order_results(results)
    sents = []
    for docid in list(ordered_results['2'].keys())[:50]:
        text = corpus[docid]['text']
        sents.append(text)
    sbert_model = SentenceTransformer(model_name)
    embeddings = sbert_model.encode(sents, normalize_embeddings=True)
    with open("punishment_pre_embeddings.pkl", "wb") as f:
        pickle.dump((sents, embeddings), f)
    # for docid in list(ordered_results['2'].keys())[:10]:
    #     try:
    #         print(f'Doc ID:{docid} - {corpus[docid]}- {qrels["2"][docid]}')
    #     except:
    #         print(f'Doc ID:{docid} - {corpus[docid]}- 0')
    # found_irrels, missed_rels = missing_rels(ordered_results, symptom)

    
    # with open('../error_analysis/irrels_found_'+symptom+'_scores.txt', 'w+') as f:
    #     for rel in found_irrels:
    #         print(f"{rel} - {corpus[rel]['text']}")
    #         f.write(corpus[rel]['text']+"-"+str(ordered_results['4'][rel])+'\n')
        
    # with open('../error_analysis/missed_rels_'+symptom+'_scores.txt', 'w+') as f:
    #     for rel in missed_rels:
    #         print(f"{rel} - {corpus[rel]['text']}")
    #         f.write(corpus[rel]['text']+"-"+str(ordered_results['4'][rel])+'\n')

    # for k,v in ordered_results.items():
    #     top_k=0
    #     for sid, score in v.items():
    #         top_k+=1
    #         sentence = corpus[sid]['text']
    #         try:
    #             rel = qrels[str(k)][sid]
    #         except:
    #             rel=0
    #         error_analysis.loc[len(error_analysis)] = [k, sid, sentence, rel]
    #         if top_k==10:
    #             break
    # error_analysis.to_csv(f'../error_analysis/pos/{symptom}_nli.csv', index=False)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symptom", nargs='?', default="punishment feelings") ### With this param we select the kind of query: only the BDI item tite, the firs question, etc.
    args = parser.parse_args()
    corpus,queries,qrels = load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/options/queries/queries_"+str(args.symptom)+".jsonl", "../dataset_format_beir/options/qrels/qrels_"+str(args.symptom)+".tsv")
    print(len(corpus))
    
    # contriever_model_name = "facebook/contriever" #### contriever
    # word_embedding_model = sentence_transformers.models.Transformer(contriever_model_name)
    # pooling_model = sentence_transformers.models.Pooling(
    #     word_embedding_model.get_word_embedding_dimension(),
    #     pooling_mode_mean_tokens=True,
    #     pooling_mode_cls_token=False,
    #     pooling_mode_max_tokens=False
    # )
    # model_name = sentence_transformers.SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model_name = "all-mpnet-base-v2" # './models/contr-bdi-sim-model' #  
    ndcg, _map, recall, precision = evaluate_retrieval(model_name, corpus, queries, qrels)
    # row = ["cont-bdi-sim", args.symptom, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],\
    #      recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    # print(row)
    # with open("../custom_sols/output.csv",'a+') as f:
    #     writer_object = writer(f)
    #     writer_object.writerow(row)
    #     f.close()
