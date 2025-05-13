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


def format_24(
    ddir:str # dir file containing the corpus and the majority qrels
)-> None:
    pools = pd.read_csv(ddir+'/pools_2024_t3.csv')
    pools["query_num"]+=1
    qrels = pd.read_csv(ddir+'/majority_erisk_2024.csv',  sep='\t', names=["query", "Q0", "doc_id", "rel_score"])
    with open('../dataset_format_beir/2024/sentences_only_text.jsonl', 'w') as fp: ## Save sentences file
        for _, row in pools.iterrows():
            doc = {}
            doc["id"] = row["doc_id"]
            doc["text"] = row["text"]
            doc["title"] = ""
            fp.write(json.dumps(doc)+'\n') 
    
    qrels = qrels.drop(columns=['Q0'])
    for _,g in qrels.groupby('query'):
        ### This part generates one query file per BDI item options
        query_id = g.iloc[0,:]["query"]
        print(query_id)
        qs = pools[pools["query_num"]==query_id].iloc[0,:]["query_str"].split('\n')
        qs = qs[:-1]  ### removes last trail line
        title = qs[0].replace(':', '').lower()
        query_nb = len(qs[1:]) ### variable number of options per bdi symptom

        qrels_symptom = pd.DataFrame()
        g["query"] = [1]*len(g)
        for i in range(query_nb):
            duplicate = g.copy()
            duplicate.iloc[:,0] = duplicate.iloc[:,0] + i
            qrels_symptom = qrels_symptom.append(duplicate)

        qrels_symptom['query'] = qrels_symptom['query'].astype(str)
        qrels_symptom.to_csv('../dataset_format_beir/2024/options/qrels/qrels_'+title+'.tsv', sep='\t', header=False, index=False)


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
    format_24("../original_dataset/2024")
    # format_options("../original_dataset/pools_docnos.json", "../original_dataset/qrels-majority.txt")
