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
            # qs = pool["query"].split('\n')
            query["_id"], query["text"] = str(count), pool["query"] # qs[0].replace(':', '') # update needed: current version only with BDI item title
            queries.append(query)
            count+=1
            
            for pair in pool["pool_list"]:
                doc = {}
                doc["_id"], doc["text"], doc["title"] = pair[0], pair[1], ""
                sentences.append(doc)

    with open('../dataset_format_beir/queries_BDI_item.jsonl', 'w') as fp: ## Save queries file
        for query in queries:
            fp.write(json.dumps(query)+'\n')

    with open('../dataset_format_beir/sentences.jsonl', 'w') as fp: ## Save sentences file
        for doc in sentences:   
            fp.write(json.dumps(doc)+'\n')


'''
This method takes the pools file and the qrels file in TREC format and generates the qrels per symptom, queries per symptom and corpus file

Args: 
    pool_path: path to the pool file
    qrels_path: path to the qrels file in TREC format
Return:
    None (just saves the new files in BEIR format)

'''
def format_options(
    pool_path: str,
    qrels_path:str
)-> None:
    sentences = []
    with open(pool_path) as f: ### Read original file
        data = json.load(f)

        symptom_nb = 1
        for pool in data["pools"]:
            queries=[]
            qs = pool["query"].split('\n')
            qs = qs[:-1]  ### removes last trail line
            title = qs[0].replace(':', '').lower()
            
            ### This part generates one query file per BDI item options
            query_nb = 1
            for option in qs[1:]:
                query = {}
                query["_id"], query["text"] = str(query_nb), option.replace('\t','')
                queries.append(query)
                query_nb+=1
            
            with open('../dataset_format_beir/options/queries/queries_'+title+'.jsonl', 'w') as fp: ## Save queries file
                for query in queries:
                    fp.write(json.dumps(query)+'\n')
            
            ### This part creates a replicated qrels file for each query option
            qrels = pd.read_csv(qrels_path, sep='\t', names=["query", "Q0", "sentence_id", "label"])
            qrels = qrels.drop(columns=['Q0'])
            

            qrels_symptom = pd.DataFrame()
            symptom_df = qrels[qrels["query"]==symptom_nb]
            symptom_df["query"] = [1]*len(symptom_df)
            print(query_nb)
            for i in range(query_nb-1):
                duplicate = symptom_df.copy()
                duplicate.iloc[:,0] = duplicate.iloc[:,0] + i
                qrels_symptom = qrels_symptom.append(duplicate)

            qrels_symptom['query'] = qrels_symptom['query'].astype(str)
            qrels_symptom.to_csv('../dataset_format_beir/options/qrels/qrels_'+title+'.tsv', sep='\t', header=False, index=False)

            ### This parts saves the sentences (common part to previous query sets)
            for pair in pool["pool_list"]:
                doc = {}
                doc["_id"], doc["text"], doc["title"] = pair[0], pair[1], ""
                sentences.append(doc)

            symptom_nb+=1

        

    with open('../dataset_format_beir/sentences.jsonl', 'w') as fp: ## Save sentences file
        for doc in sentences:   
            fp.write(json.dumps(doc)+'\n')

if __name__=="__main__":
    # format_qrels("../original_dataset/qrels-majority.txt")
    # format_data("../original_dataset/pools_docnos.json")
    format_options("../original_dataset/pools_docnos.json", "../original_dataset/qrels-majority.txt")
