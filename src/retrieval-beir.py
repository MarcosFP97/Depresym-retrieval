from beir.datasets.data_loader import GenericDataLoader

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


if __name__=="__main__":
    load_custom_data("../dataset_format_beir/sentences.jsonl", "../dataset_format_beir/queries.jsonl", "../dataset_format_beir/qrels.tsv")