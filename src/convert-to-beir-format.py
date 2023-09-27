from tkinter.font import names
import pandas as pd
import json

'''
This method takes the qrels file in TREC format and saves the qrels in a BEIR acceptable format

Args: 
    path: path to the qrels file in TREC format
Return:
    None (just saves the new file in BEIR format)

'''
def format_qrels(
    path: str
) -> None:
    qrels = pd.read_csv(path, sep='\t', names=["query", "Q0", "sentence_id", "label"])
    qrels = qrels.drop(columns=['Q0'])
    qrels.to_csv('../dataset_format_beir/qrels.txt', sep='\t', header=False, index=False)

'''
This method takes the Depresym pool file and generates a file of queries and another for sentences in a BEIR acceptable format

Args: 
    path: path to the Depresym pool file
Return:
    None (just saves the new files in BEIR format)

'''
def format_data(
    path: str
)-> None:
    queries, sentences = {}, {}
    with open(path) as f: ### Read original file
        data = json.load(f)

        count = 1
        for pool in data["pools"]:
            queries[count] = pool["query"]
            count+=1
            
            for pair in pool["pool_list"]:
                sentences[pair[0]] = pair[1]

    with open('../dataset_format_beir/queries.json', 'w') as fp: ## Save queries file
        json.dump(queries, fp)

    with open('../dataset_format_beir/sentences.json', 'w') as fp: ## Save sentences file
        json.dump(sentences, fp)

if __name__=="__main__":
    format_qrels("../original_dataset/qrels-majority.txt")
    format_data("../original_dataset/pools_docnos.json")
