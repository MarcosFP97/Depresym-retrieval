import pandas as pd 
from scipy.stats import wilcoxon

df = pd.read_csv('../custom_sols/output_2024.csv', names=["method", "symptom", "map10", "map100", "map1000", "p10", "p100", "p1000", "r10", "r100", "r1000", "ndcg10", "ndcg100", "ndcg1000"])
baseline = df[df["method"]=="pos"]['r100'].values
print(f'{baseline.mean()}')
bdi_rand = df[df["method"]=="qwen-sim"]['r100'].values
print(f'{bdi_rand.mean()}')
stat, p_value = wilcoxon(baseline, bdi_rand)
print(f'p-value for R@100: {p_value}')

baseline = df[df["method"]=="pos"]['ndcg10'].values
print(f'{baseline.mean()}')
bdi_rand = df[df["method"]=="qwen-sim"]['ndcg10'].values
print(f'{bdi_rand.mean()}')
stat, p_value = wilcoxon(baseline, bdi_rand)
print(f'p-value for NDCG@10: {p_value}')

baseline = df[df["method"]=="pos"]['ndcg1000'].values
print(f'{baseline.mean()}')
bdi_rand = df[df["method"]=="qwen-sim"]['ndcg1000'].values
print(f'{bdi_rand.mean()}')
stat, p_value = wilcoxon(baseline, bdi_rand)
print(f'p-value for NDCG@1000: {p_value}')