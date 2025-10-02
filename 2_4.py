from tqdm import tqdm
import time
import json
import datasets
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM


eval_set = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")
grouped_datasets = {}
for example in eval_set['train']:
    dataset = example["dataset"]
    if dataset not in grouped_datasets:
        grouped_datasets[dataset] = []
    grouped_datasets[dataset].append(example)


datasets_names = ["piqa", "OpenbookQA", "CommonsenseQA"]
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("/home/users/ntu/yixuan02/models/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/users/ntu/yixuan02/models/Qwen2.5-3B-Instruct").to(device)

# Define a function for batch inference
def infer_llm_batch(samples, batch_size=32):
    all_prompts = []
    for sample in samples:
        question = sample["question"]
        options = sample["options"]
        prompt = f"{question}\n"
        for option in options:
            prompt += f"{option['label']}: {option['text']}\n"

        prompt = """What is the capital of France?
        A: London
        B: Paris
        C: Melbourne
        D: USA
        Let's think step by step and answer in **So the answer is A/B/C/D**: France is a country in Europe, and Paris is its capital city. So the answer is B.
        """ + prompt
        prompt += "Let's think step by step and answer in **So the answer is A/B/C/D**: "
        all_prompts.append(prompt)
    
    all_answers = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches"):
        batch_prompts = all_prompts[i:i+batch_size]
        
        # 批量tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                num_return_sequences=1,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )
        
        # 批量解码和答案提取
        for j, output in enumerate(outputs):
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            
            # 调试：打印前几个样本的输出
            if len(all_answers) < 3:
                print(f"\n=== DEBUG: Sample {len(all_answers)+1} ===")
                print(f"Decoded output: {decoded_output[-200:]}")  # 只打印最后200字符
                print("="*50)
            
            import re
            # 尝试多种匹配模式
            patterns = [
                r'So the answer is ([ABCD])',
                r'answer is ([ABCD])',
                r'Answer: ([ABCD])',
                r'([ABCD])\s*[.!]?\s*$',  # 行末的字母
                r'[^A-Z]*([ABCD])[^A-Z]*$',  # 最后一个大写字母
            ]
            
            answer = ""
            for pattern in patterns:
                match = re.search(pattern, decoded_output.upper())
                if match:
                    answer = match.group(1).strip()
                    break
            
            # 如果还是没找到，就找所有ABCD字母，取最后一个
            if not answer:
                letters = re.findall(r'[ABCD]', decoded_output.upper())
                answer = letters[-1] if letters else ""
            
            all_answers.append(answer)
    
    return all_answers

# Define the evaluation function
def evaluate_samples(samples):
    correct = 0
    total = len(samples)

    s = time.time()
    # 使用批量推理
    predictions = infer_llm_batch(samples, batch_size=32)
    
    # 计算准确率
    for i, sample in enumerate(samples):
        if predictions[i] == sample["answerKey"]:
            correct += 1
    
    e = time.time()
    print("Time:",e - s)
    accuracy = correct / total * 100
    return predictions, accuracy

# Evaluate the samples
for dataset_name in datasets_names:
    print(f"\n=== Evaluating {dataset_name} ===")
    predictions, accuracy = evaluate_samples(grouped_datasets[dataset_name])
    for i, sample in enumerate(grouped_datasets[dataset_name]):
        print("="*100)
        print(f"Question: {sample['question']}")
        print("-"*50)
        print(f"Predicted Answer: {predictions[i]}")
        print("-"*50)
        print(f"Correct Answer: {sample['answerKey']}\n")
        print("="*100)
    print(f"Accuracy: {accuracy:.2f}%")
    
# # Print results
# for i, sample in enumerate(grouped_datasets['CommonsenseQA']):
#     print(f"Question: {sample['question']}")
#     print(f"Predicted Answer: {predictions[i]}")
#     print(f"Correct Answer: {sample['answerKey']}\n")