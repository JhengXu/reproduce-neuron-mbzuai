import os

# 批量设置环境变量
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/.catch'
os.environ['HF_CACHE_DIR'] = '/root/autodl-tmp/.catch'
os.environ['HF_HOME'] = '/root/autodl-tmp/.catch/huggingface'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp/.catch/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = 'your HF_TOKEN'
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import regex as re
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import random
from datasets import load_dataset
from evaluate import load
import metrics.n_gram
import metrics.Information_Entropy

# <!-- ADD --> 新增配置
GENERATE_KWARGS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}

# Set random seed
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)


# Load model
def prepareModel(modelName):
    if 'gemma' in modelName or 'phi' in modelName or 'llm-jp' in modelName:
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModelForCausalLM.from_pretrained(
            modelName,
            device_map="auto"
        )
    elif 'pythia' in modelName:
        from transformers import GPTNeoXForCausalLM, AutoTokenizer
        model = GPTNeoXForCausalLM.from_pretrained(modelName, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(modelName)
    elif 'swallow' in modelName:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModelForCausalLM.from_pretrained(
                    modelName, torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True, device_map="auto")
    elif 'Llama-3' in modelName:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModelForCausalLM.from_pretrained(
            modelName,
            device_map="auto"
        )
    return tokenizer, model

class OutputInspector:
  def __init__(self, targetLayer):
      self.layerOutputs = []
      self.featureHandle=targetLayer.register_forward_hook(self.feature)

  def feature(self,model, input, output):
      self.layerOutputs.append(output.detach().cpu())

  def release(self):
      self.featureHandle.remove()

def countOverlap(text, query):
    return len(re.findall(query, text, overlapped=True))

def getFirstAppearingIdx(text, query):
    return re.finditer(query, text).__next__().start(0)

def detectRepetition(line, n, r, k):
    SEP = ' '
    line_str = SEP.join(map(str, line))
    for i in range(0, len(line)-n+1):
        ngram = line[i:i+n]
        ngramStr = SEP.join(map(str, ngram))
        lineRange = line[max(0, i+n-r):i+n]
        lineRangeStr = SEP.join(map(str, lineRange))
        countRepInRange = countOverlap(lineRangeStr, ngramStr)

        if k <= countRepInRange:
            try:
                firstPosition = getFirstAppearingIdx(lineRangeStr, ngramStr)
                firstPosition = len(lineRangeStr[:firstPosition].split()) 
                firstPosition += max(0, i+n-r)

                lineRangeStr4second = SEP.join(map(str, line[firstPosition+1:i+n]))
                secondPosition = getFirstAppearingIdx(lineRangeStr4second, ngramStr)
                secondPosition = len(lineRangeStr4second[:secondPosition].split())
                secondPosition += firstPosition + 1

                lineRangeStr4third = SEP.join(map(str, line[secondPosition+1:i+n]))
                thirdPosition = getFirstAppearingIdx(lineRangeStr4third, ngramStr)
                thirdPosition = len(lineRangeStr4third[:thirdPosition].split())
                thirdPosition += secondPosition + 1

                if (secondPosition - firstPosition) == (thirdPosition - secondPosition):
                    return ngram, firstPosition, secondPosition, thirdPosition
            except:
                pass
    return [], -1, -1, -1


def load_custom_dataset(dataset_path):
    ds = load_dataset('json', data_files=dataset_path)
    return ds

def preprocess_sample(sample, dataset_type):
    if dataset_type == "Academic":
        response_edited = re.search(r'<question>(.*?)</question>', sample['response'])
        return response_edited.group(1).strip() if response_edited else None
    elif dataset_type == "Diversity":
        return sample['question']
    elif dataset_type == "natural":
        return sample['query']
    return None


def generate_dataset(dataset,save_path,tokenizer,model):
    
    ds = load_dataset(dataset)
    repetition_dataset = []
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(500), desc='Processing dataset'):
            try:
                    # 数据预处理
                if 'Diversity' in dataset:
                    prompt = ds['train']['question'][i]
                if 'Academic' in dataset:
                    response = ds['train']['response'][i]
                    response_edited = re.search(r'<question>(.*?)</question>', response)
                    prompt = response_edited.group(1).strip()  # 打印匹配到的 question 内容
                if 'natural' in dataset:
                    prompt = ds['train']['query'][i]
                model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                output = model.generate(**model_inputs, **GENERATE_KWARGS)
                decoded = tokenizer.decode(output[0])
                decoded_edited = decoded.split("\n", 1)[-1].strip()

                # 检测重复
                token_ids = output[0].tolist()
                ngram, fp, sp, tp = detectRepetition(token_ids, n=8, r=100, k=3)

                # # 计算指标
                # perplexity_value = None
                # if decoded_edited:
                #     try:
                #         results = perplexity.compute(predictions=[decoded_edited], model_id=MODEL_PATH)
                #         perplexity_value = round(results["perplexities"][0], 2)
                #     except Exception as e:
                #         print(f"Perplexity error: {e}")

                # 保存结果
                result = {
                    "prompt": prompt,
                    "output": decoded_edited,
                    "token_ids": token_ids,
                    "repetition": {
                        "detected": bool(ngram),
                        "positions": [fp, sp, tp],
                        "ngram": tokenizer.decode(ngram) if ngram else None
                    }
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

                # 收集重复样本
                if result['repetition']['detected']:
                    repetition_entry = {
                        "prompt": prompt,
                        "generatedIds": token_ids,
                        "firstPosition": fp,
                        "secondPosition": sp,
                        "thirdPosition": tp,
                        "ngramIds": ngram,
                        "ngramTokens": tokenizer.decode(ngram),
                    }
                    repetition_dataset.append(repetition_entry)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
    
    return repetition_dataset

def getActs(model, tokenizer, inputIds):
    model.eval()
    with torch.no_grad():
        from transformers import (
            GPT2LMHeadModel,
            LlamaForCausalLM,
            GemmaForCausalLM,
            Gemma2ForCausalLM  # 新增对 Gemma2 的支持
        )

        actInspectors = []

        # GPT-2 分支
        if isinstance(model, GPT2LMHeadModel):
            actInspectors = [OutputInspector(layer.mlp.act) 
                            for layer in model.transformer.h]
        
        # LLaMA 分支
        elif isinstance(model, LlamaForCausalLM):
            actInspectors = [OutputInspector(layer.mlp.act_fn) 
                           for layer in model.model.layers]
        
        # 同时支持 Gemma 和 Gemma2 的分支
        elif isinstance(model, (GemmaForCausalLM, Gemma2ForCausalLM)):
            actInspectors = [OutputInspector(layer.mlp.act_fn) 
                           for layer in model.model.layers]
        
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")

        # 后续操作保持不变
        input_ids = torch.LongTensor([inputIds]).to(model.device)
        outputs = model(input_ids)
        
        for actInspector in actInspectors:
            actInspector.release()

        acts = torch.cat([torch.cat(actInspector.layerOutputs, dim=1) 
                        for actInspector in actInspectors], dim=0).transpose(0,1)
    return acts

def getAveragedActivations(data, model, tokenizer, maxRange=50, position='first'):
    repPosition = '%sPosition'%position

    normalActs = None
    repetiActs = None
    normalTotalPoints = 0
    repetiTotalPoints = 0

    for line in tqdm(data):
        inputIds = line['generatedIds']
        acts = getActs(model, tokenizer, inputIds)
        startingPoint = line[repPosition] - 1
        normalRange = list(range(max(0, startingPoint-maxRange), startingPoint))
        repetiRange = list(range(startingPoint, min(len(inputIds), startingPoint + maxRange)))

        normalTotalPoints += len(normalRange)
        repetiTotalPoints += len(repetiRange)

        na = acts[normalRange].sum(dim=0)
        ra = acts[repetiRange].sum(dim=0)

        if normalActs is None:
            normalActs = na
        else:
            normalActs += na

        if repetiActs is None:
            repetiActs = ra
        else:
            repetiActs += ra

    normalActs /= normalTotalPoints
    repetiActs /= repetiTotalPoints

    return normalActs, repetiActs

def findNeurons(data, model, tokenizer, maxRange=50, position='second'):
    normalActs, repetiActs = getAveragedActivations(data, model, tokenizer, maxRange, position)
    diff = repetiActs - normalActs
    ranks = torch.argsort(diff.flatten(), descending=True)
    width = diff.shape[1]
    sortedNeurons = []
    for r in ranks:
        neuron = (int(r // width), int(r % width))
        info = {
            'neuron': neuron,
            'normalActs':  normalActs[neuron].tolist(),
            'repetitionActs': repetiActs[neuron].tolist(),
            'diffs': diff[neuron].tolist()
        }
        sortedNeurons.append(info)
    return sortedNeurons


class Activator():
    def __init__(self, targetLayer, neuronIds, mode, lastN=0):
        self.neuronIds = neuronIds

        assert mode in ['last', 'all', 'lastN'], 'mode should be last or all'
        self.mode = mode
        self.lastN = lastN

        self.outputHandle = targetLayer.register_forward_hook(self.activate)

    def activate(self,model, input, output):
        if self.mode == 'last':
          output[0, -1, self.neuronIds] += 1
        elif self.mode == 'all':
          output[0, :, self.neuronIds] += 1
        elif self.mode == 'lastN':
          output[0, -self.lastN:, self.neuronIds] += 1
        else:
          print(f'{self.mode=} cannot be recognized')
          pass
        return output

    def release(self):
        self.outputHandle.remove()

class Deactivator():
    def __init__(self, targetLayer, neuronIds, mode, lastN=0):
        self.neuronIds = neuronIds

        assert mode in ['last', 'all', 'lastN'], 'mode should be last or all'
        self.mode = mode
        self.lastN = lastN

        self.outputHandle = targetLayer.register_forward_hook(self.deactivate)

    def deactivate(self,model, input, output):
        if self.mode == 'last':
          output[0, -1, self.neuronIds] *= 0
        elif self.mode == 'all':
          output[0, :, self.neuronIds] *= 0
        elif self.mode == 'lastN':
          output[0, -self.lastN:, self.neuronIds] *= 0
        else:
          print(f'{self.mode=} cannot be recognized')
          pass
        return output

    def release(self):
        self.outputHandle.remove()

def convertNeuronsToDict(neurons):
    layer2neurons = {}
    for fn in neurons:
        i, j = fn
        if i not in layer2neurons:
            layer2neurons[i] = []
        layer2neurons[i].append(j)
    return layer2neurons
def generateWithIntervention(model, tokenizer, initialInput, neurons, mode):
    model.eval()
    
    # 动态导入模型类用于类型检查
    from transformers import (
        GPT2LMHeadModel,
        LlamaForCausalLM,
        GemmaForCausalLM,
        Gemma2ForCausalLM
    )

    # 初始化干预模式
    INTERV = Activator if mode == 'activate' else Deactivator
    layer2neurons = convertNeuronsToDict(neurons)

    # 根据模型类型选择不同的干预路径
    acts = []
    if isinstance(model, GPT2LMHeadModel):  # GPT-2
        # GPT-2的激活函数在mlp模块的act属性
        acts = [
            INTERV(layer.mlp.act, layer2neurons[i], 'last')
            for i, layer in enumerate(model.transformer.h)
            if i in layer2neurons
        ]
    
    elif isinstance(model, LlamaForCausalLM):  # LLaMA-3
        # LLaMA的激活函数在mlp.act_fn
        acts = [
            INTERV(layer.mlp.act_fn, layer2neurons[i], 'last')
            for i, layer in enumerate(model.model.layers)
            if i in layer2neurons
        ]
    
    elif isinstance(model, (GemmaForCausalLM, Gemma2ForCausalLM)):  # Gemma 系列
        # Gemma激活函数位置与LLaMA一致
        acts = [
            INTERV(layer.mlp.act_fn, layer2neurons[i], 'last')
            for i, layer in enumerate(model.model.layers)
            if i in layer2neurons
        ]
    
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")

    # 生成配置保持不变
    initialInput = torch.LongTensor([initialInput]).to(model.device)
    generationConfigGreedy = GenerationConfig(
        max_new_tokens=160,  # 调整为更合理的长度
        do_sample=False,
        eos_token_id=model.config.eos_token_id
    )
    
    # 执行带干预的生成
    with torch.inference_mode():
        additionalOutputs = model.generate(
            initialInput,
            generation_config=generationConfigGreedy
        )

    # 释放所有干预器
    for a in acts:
        a.release()

    # 检测重复模式
    output_ids = additionalOutputs[0].tolist()
    ngram, *positions = detectRepetition(output_ids, n=10, r=100, k=3)

    return output_ids, ngram, *positions



def compute_metrics(input_sentence, metric):
    if metric == 'rs':
        return metrics.n_gram.calculate_rep_n(input_sentence, 1)
    if metric == 'ie':
        return metrics.Information_Entropy.calculate_ngram_entropy(input_sentence, 1)
    

def experiment_wrapper(save_path,repetitionDataset, model, tokenizer,model_path):  
    
    # 执行神经元分析
    sorted_neurons = findNeurons(repetitionDataset, model, tokenizer)
    
    # 执行干预实验
    intervention_results = []
    for sample in tqdm(repetitionDataset[:500]):
        try:
            # 原始生成
            orig_output = tokenizer.decode(sample['generatedIds'])
            # 干预生成
            intervened_ids, _, _, _, _ = generateWithIntervention(
                model, tokenizer, 
                sample['generatedIds'][:sample['secondPosition']],
                [n['neuron'] for n in sorted_neurons[:1000]],
                'deactivate'
            )
            intervened_output = tokenizer.decode(intervened_ids)
            print(intervened_output)
            rs_score1 = compute_metrics(orig_output, 'rs')
            ie_score1 = compute_metrics(orig_output, 'ie')
            rs_score = compute_metrics(intervened_output, 'rs')
            ie_score = compute_metrics(intervened_output, 'ie')
            perplexity = load("perplexity",module_type="metric")
            results = perplexity.compute(model_id=model_path,predictions=intervened_output)
            intervention_results.append({
                "original": orig_output,
                "intervened": intervened_output,
                "repeat_origin": rs_score1,
                "entropy_origin": ie_score1,
                "repeat score": rs_score,
                "information entropy": ie_score,
                "perplexity": round(results["perplexities"][0], 2),
            })
        except Exception as e:
            print(f"Intervention failed: {e}")
    
    # 将结果写入JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(intervention_results, f, ensure_ascii=False, indent=2)
    
    return intervention_results  # 保留返回值（按原始逻辑）

def test(model_path,dataset,save_path):
    
    # 加载模型
    tokenizer, model = prepareModel(model_path)
    model.eval()

    # 加载指标和数据集
    perplexity = load("perplexity", module_type="metric")
    repetitionDataset = generate_dataset(dataset,save_path,tokenizer,model)

    # 查找神经元
    sortedNeurons = findNeurons(repetitionDataset, model, tokenizer, maxRange=50)
    
    # 打印前500个神经元
    for line in sortedNeurons[:500]:
        print(line)

    # 计算最大层号
    maxLayerNum = max([neuron['neuron'][0] for neuron in sortedNeurons]) if sortedNeurons else 0

    # 初始化坐标数据
    xs = [i/maxLayerNum for i in range(maxLayerNum+1)] if maxLayerNum !=0 else []
    ys = [0] * (maxLayerNum + 1) if maxLayerNum !=0 else []

    size = int(len(sortedNeurons)*0.005)
    print(f'{size=}')

    # 关键修复：循环内代码需要正确缩进！
    for neuron in sortedNeurons[:size]:
        if neuron['diffs'] < 0:
            continue
        # 以下两行必须缩进到循环内部
        layerPosition = neuron['neuron'][0]/maxLayerNum
        ys[xs.index(layerPosition)] += 1  # 确保 xs 中存在 layerPosition

    # 运行实验
    intervention_results = experiment_wrapper(save_path,repetitionDataset, model, tokenizer,model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, choices=["google/gemma-2-2b","meta-llama/Llama-3.1-8B"])
    parser.add_argument('--dataset', type=str, choices=["DisgustingOzil/Academic_dataset_ShortQA", "YokyYao/Diversity_Challenge", "sentence-transformers/natural-questions"])
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    test(model_path=args.model_path,dataset=args.dataset,save_path=args.save_path)

if __name__ == "__main__":
    main()
