from datasets import load_dataset
import pandas as pd
import time
from extract import extractive_summary
from abstract import abstractive_summary
from tqdm import tqdm


print("Loading Dataset. This may take a while...")
ds = load_dataset("ccdv/arxiv-summarization", "document")
df = pd.DataFrame(ds['train'])

# Test Extractive Method 10,000 Times
#TODO: Add accuracy metric test in the loop
samples = 1000
s_time = time.time()
for i in tqdm(range(samples), desc="Running Extractive Method Tests"):
    extractive_summary(df['article'][i])
e_time = time.time()

program_time = e_time - s_time
print(program_time)

# Test Abstractive Method 10 Times
#TODO: Add accuracy metric test in the loop
samples = 5
s_time = time.time()
for i in tqdm(range(samples), desc="Running Abstractive Method Tests"):
    abstractive_summary(df['article'][i])
e_time = time.time()

program_time = e_time - s_time
print(program_time)