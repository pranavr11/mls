"""
This file contains all the necessary information for running baseline benchmarks of small transformer
models with full attention on long-context tasks. The initial baseline will use pretrained weights from 
GPT-2 and Pythia-160M, running on the PG-19 (Project Gutenberg books) and SCROLLS (a set of 
long-context reasoning benchmarks) datasets. 
"""

from dotenv import load_dotenv
import os

load_dotenv()

"""
Step 1: Download the data from PG-19 and SCROLLS. Store it in the 'data' directory. 

Note: For now, we only load the gov_report task of SCROLLS (since I don't have enough local disk space)
"""
print("Running baselines.py")
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

cache_dir = "./data"
# pg19 = load_dataset("pg19", cache_dir=cache_dir)
configs = [
    c for c in get_dataset_config_names("tau/scrolls")
    if c != "narrative_qa"
]
print(configs)
scrolls = {}

for name in tqdm(configs, desc="Loading SCROLLS"):
    print(name)
    scrolls[name] = load_dataset(
        "tau/scrolls",
        name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
scrolls = load_dataset("tau/scrolls", "gov_report", cache_dir=cache_dir)


"""
Step 2:
Load the pretrained model weights for Pythia-160M and GPT-2
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

# print(pg19)
print(scrolls)


"""
Step 3: Benchmark how well each model predicts the data and how expensive it is to do so. 
+ Visualize additional metrics such as attention maps 

First, we benchmark Pythia-160M, then GPT-2. 
"""

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

"""
The gov_report task is to take in a government agency document and summarize it. 
This is presented as a sequence-to-sequence task, where the model takes in [DOCUMENT] + [SEP] + [SUMMARY]
and predicts the conditional probability of each token in the summary given all previous summary 
tokens and document tokens. 

The original SCROLLS paper aggregates ROUGE-1 (unigram overap) F1, ROUGE-2 (bigram overlap) F1 and 
ROUGE-L (longest overlapping subsequence) F1 to produce a single final ROUGE score. 


F1 = 2 * (precision * recall) / (precision + recall)
Calculating the harmonic mean forces both precision and recall to be high. 

We calculate the ROUGE score of Pythia-160M on one item in the GovReport validation dataset
"""

example = scrolls["validation"][0] 
inputs = tokenizer(example["input"], return_tensors="pt", truncation=True, max_length=1024)
output_ids = model.generate(**inputs, max_new_tokens=200)
prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
reference = example["output"]

from evaluate import load
print("Computing ROUGE scores: ")
rouge = load("rouge")
scores = rouge.compute(
    predictions=[prediction],
    references=[reference]
)

"""
Step 3: 
Now we compute the latency of Pythia-160M on one example from the SCROLLS/GovReport dataset
run k trials, take the average
"""
import time

k = 5
latencies = []
tokens_generated = None

# warmup
_ = model.generate(**inputs, max_new_tokens=200)

for _ in range(k):
    start = time.time()
    out = model.generate(**inputs, max_new_tokens=200)
    end = time.time()

    latencies.append(end - start)

    if tokens_generated is None:
        tokens_generated = out.shape[1] - inputs["input_ids"].shape[1]

avg_latency = sum(latencies) / k
tokens_per_sec = tokens_generated / avg_latency

print("Avg Latency:", avg_latency)
print("Tokens/sec:", tokens_per_sec)

print(scores)


"""
Step 4:
Next, we compute the attention map for that one such example
"""
import torch
import matplotlib.pyplot as plt


model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager"
)
model.eval()

example = scrolls["validation"][0]

viz_inputs = tokenizer(
    example["input"],
    return_tensors="pt",
    truncation=True,
    max_length=1024,   # keep small enough to visualize clearly
)

with torch.no_grad():
    outputs = model(
        **viz_inputs,
        output_attentions=True
    )

# attentions is a tuple: one tensor per layer
# each tensor has shape [batch_size, num_heads, seq_len, seq_len]
layer_idx = 0
head_idx = 0

attn = outputs.attentions[layer_idx][0, head_idx].cpu()
tokens = tokenizer.convert_ids_to_tokens(viz_inputs["input_ids"][0])

plt.figure(figsize=(10, 8))
plt.imshow(attn, aspect="auto")
plt.colorbar()
plt.title(f"Attention Map | Layer {layer_idx} | Head {head_idx}")
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
plt.yticks(range(len(tokens)), tokens, fontsize=6)
plt.tight_layout()
plt.show()


"""



"""


"""
The issue with our current algorithm is that we compute the top k attention scores per window — this means that
only tokens which are important in that window have a possibility of being remembered, even when they might 
be important globally. So instead of purely relying on our local sliding window attention scores, we should 
have a "global attention" metric that is trained to recognize globally important tokens even when they may not 
be important in the current window. The examples of this can come from computing full attention, identifying such 
tokens, and having some model analyze it for patterns. 
"""