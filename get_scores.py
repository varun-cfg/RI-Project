import json
import math

with open("scores_llama.json") as f:
    scores = json.load(f)

samples = []
for task in scores:
    task_sum = 0
    mean = 3.6188
    for j in scores[task]:
        samples.append(j)

samples = [5,5,5,4,4,4,1,3,3,5,4,2,3,5,1,4,5,5,5,4,5,2,5,5,4,4,4,4,5,5,2,3,5,4,3]

std = 0.0
for i in samples:
    std+=(i-mean)**2

std = math.sqrt(std/len(samples))
print("std: ",std)
print("num samples: ",len(samples))

