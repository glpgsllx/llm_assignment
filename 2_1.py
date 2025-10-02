from tqdm import tqdm
import time
import json
import datasets
from transformers import AutoTokenizer,AutoModelForCausalLM


eval_set = datasets.load_dataset("Open-Style/Open-LLM-Benchmark", "questions")
grouped_datasets = {}
for example in eval_set['train']:
    dataset = example["dataset"]
    if dataset not in grouped_datasets:
        grouped_datasets[dataset] = []
    grouped_datasets[dataset].append(example)


datasets_names = ["CommonsenseQA", "OpenbookQA", "piqa"]
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("/home/users/ntu/yixuan02/models/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/users/ntu/yixuan02/models/Qwen2.5-3B-Instruct").to(device)

# Define a function for inference
def infer_llm(sample):
    question = sample["question"]
    options = sample["options"]
    prompt = f"{question}\n"
    for option in options:
        prompt += f"{option['label']}: {option['text']}\n"

    ### You can change the prompt to to suit the model you are using.
    # Example:
    # Answer in A/B/C/D:
    # Answer in a single word or phrase:

    prompt += "Answer in A/B/C/D: "

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=len(inputs["input_ids"][0]) + 5, ### You may refer to the max_new_tokens parameter to speed up inference.
        num_return_sequences=1,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded_output.split("Answer:")[-1].strip()
    return answer

# Define the evaluation function
def evaluate_samples(samples):
    correct = 0
    total = len(samples)
    predictions = []

    s = time.time()
    for sample in tqdm(samples):
        predicted_answer = infer_llm(sample)
        predictions.append(predicted_answer)

        if predicted_answer == sample["answerKey"]:
            correct += 1
    e = time.time()
    print("Time:",e - s)
    accuracy = correct / total * 100
    return predictions, accuracy

# Evaluate the samples
for dataset_name in datasets_names:
    print(f"\n=== Evaluating {dataset_name} ===")
    predictions, accuracy = evaluate_samples(grouped_datasets[dataset_name])
    print(f"Accuracy: {accuracy:.2f}%")
    
# # Print results
# for i, sample in enumerate(grouped_datasets['CommonsenseQA']):
#     print(f"Question: {sample['question']}")
#     print(f"Predicted Answer: {predictions[i]}")
#     print(f"Correct Answer: {sample['answerKey']}\n")