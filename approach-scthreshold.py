import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-p", "--path", default = 5, help="number of paths to use for self consistency")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    path = int(args.path)
    portion = args.portion

    lm_utils.llm_init(model_name)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []

    with open("data/" + dataset + ".json", "r") as f:
        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]
        
        # obtain correct flags for test set

        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            # print(response)
            # print(lm_utils.answer_parsing(response))
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

        # obtain threshold of self-consistency paths for abstain

        correct_labels_dev = []
        majority_answer_paths = []
        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            answers = []
            for k in range(path):
                response = lm_utils.llm_response(original_prompt, model_name, probs=False, temperature=0.7)
                answers.append(lm_utils.answer_parsing(response))
            count_largest = 0
            key_largest = None
            for key in d["choices"].keys():
                if answers.count(key) > count_largest:
                    count_largest = answers.count(key)
                    key_largest = key
            if key_largest == correct_answer:
                correct_labels_dev.append(1)
            else:
                correct_labels_dev.append(0)
            majority_answer_paths.append(count_largest)
        
        min_error = 1e6
        best_threshold = 0
        for threshold in range(2,path):
            error = 0
            for i in range(len(correct_labels_dev)):
                if majority_answer_paths[i] < threshold:
                    if correct_labels_dev[i] == 1:
                        error += 1
                else:
                    if correct_labels_dev[i] == 0:
                        error += 1
            if error < min_error:
                min_error = error
                best_threshold = threshold
        
        # obtain abstain flags for test set
        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            answers = []
            for k in range(path):
                response = lm_utils.llm_response(original_prompt, model_name, probs=False, temperature=0.7)
                answers.append(lm_utils.answer_parsing(response))
            count_largest = 0
            key_largest = None
            for key in d["choices"].keys():
                if answers.count(key) > count_largest:
                    count_largest = answers.count(key)
                    key_largest = key
            if count_largest < best_threshold:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            
            abstain_scores.append(1 - (count_largest / path))
            
    print("------------------")
    print("Approach: scthreshold")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")