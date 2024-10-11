import os
import subprocess

for symptom in os.listdir('../dataset_format_beir/options/queries'):
    symptom = symptom.split("_")
    symptom = symptom[1].split('.')
    symptom = symptom[0]
    subprocess.run(["python", "rerank-ce.py",symptom])
