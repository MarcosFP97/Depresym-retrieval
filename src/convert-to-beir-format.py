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
    qrels['query'] = qrels['query'].astype(str)
    qrels.to_csv('../dataset_format_beir/qrels.tsv', sep='\t', header=False, index=False)

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
    queries, sentences = [], []
    with open(path) as f: ### Read original file
        data = json.load(f)

        count = 1
        for pool in data["pools"]:
            query = {}
            query["_id"], query["text"] = str(count), pool["query"]
            queries.append(query)
            count+=1
            
            for pair in pool["pool_list"]:
                doc = {}
                doc["_id"], doc["text"], doc["title"] = pair[0], pair[1], ""
                sentences.append(doc)

    with open('../dataset_format_beir/queries.jsonl', 'w') as fp: ## Save queries file
        for query in queries:
            fp.write(json.dumps(query)+'\n')

    with open('../dataset_format_beir/sentences.jsonl', 'w') as fp: ## Save sentences file
        for doc in sentences:   
            fp.write(json.dumps(doc)+'\n')

if __name__=="__main__":
    format_qrels("../original_dataset/qrels-majority.txt")
    format_data("../original_dataset/pools_docnos.json")
