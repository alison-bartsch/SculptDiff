# from transformers import LlamaForCausalLM, LlamaTokenizer

import random
import numpy as np
from sentence_transformers import SentenceTransformer, util

# have a set of pre-made prompts to randomly select for each shape goal + add in str(shape) automatically
clay_shape = "X"
prompts = [f"make an {clay_shape}", 
           f"sculpt a {clay_shape}", 
           f"please creat a {clay_shape}", 
           f"I would like a {clay_shape} sculpture",
           f"could you make a {clay_shape} for me?",
           f"I need a {clay_shape} sculpture",
           f"sculpt a {clay_shape} for me",
           f"make a {clay_shape} sculpture"]

# # initialize llama model
# tokenizer = LlamaTokenizer.from_pretrained("/output/path")
# model = LlamaForCausalLM.from_pretrained("/output/path")

# randomly select a prompt and embed with sentence transformer as a test
# prompt = random.choice(prompts).format(clay_shape=clay_shape)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = []

for prompt in prompts:
    embed = model.encode(prompt)
    print("\nSentence: ", prompt)
    embeddings.append(embed)

# print cosine similarity between all pairs of embeddings
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        print("Cosine-Similarity between embeddings {} and {}: {}".format(i, j, util.cos_sim(embeddings[i], embeddings[j])))

