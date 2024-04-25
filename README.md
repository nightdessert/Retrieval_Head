# Retrieval Head
This is the open-source code for paper:
*[Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574)*
## Retrieval Head Dectection
An algorithm that statistically calculate the retrieval score of attention heads in a transformer model.

A Single 80G GPU is enough to detect up to 50K length.
### Usage :
```python
python retrieval_head_detection.py  --model_path $path_to_model --s 0 --e 50000
```
**Currently Implemented Model Families**: 
LLama([Llama-2-7B-80K](https://huggingface.co/yaofu/llama-2-7b-80k)), Yi, Qwen, Mistrial

### Results:
All detection results are saved in "./head_score/*.json", where each head is saved in the format of 
```python
{layer-head_id: [list of retrieval scores across detections]}
```
**Directly load a results for Analysis**
```python
## load head score file, llama-2-7b-80k for example
with open('./head_score/llama-2-7b-80k.json') as file:
    head_list = json.loads(file.readline())
## use the average retrieval score and ranking
head_score_list = [([int(ll) for ll in l[0].split("-")], np.mean(l[1])) for l in head_list.items()]
head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True) 
top_retrieval_heads = [[l[0],  round(np.mean(l[1]), 2)] for l in head_score_list][:10]
print(f'\n'.join([f"Head:{l[0]},   Retrieval Score: {l[1]}"  for l in top_retrieval_heads]))
'''
Head:[16, 19],   Retrieval Score: 0.94      Head:[11, 15],   Retrieval Score: 0.92      
Head:[8, 26],    Retrieval Score: 0.8       Head:[6, 9],     Retrieval Score: 0.62        
Head:[7, 12],    Retrieval Score: 0.61      Head:[17, 22],   Retrieval Score: 0.56
Head:[11, 2],    Retrieval Score: 0.46      Head:[6, 16],    Retrieval Score: 0.44
Head:[19, 15],   Retrieval Score: 0.42      Head:[21, 30],   Retrieval Score: 0.4
'''
```
## Influence on Needle-in-a-Haystack
Coming soon
