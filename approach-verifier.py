import json
import argparse
import lm_utils
import metrics
import random
import re
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-e", "--epoch", default = 5, help="epochs of verifier training")
    argParser.add_argument("-b", "--batch", default = 32, help="batch size of verifier training")
    argParser.add_argument("-l", "--lr", default = 5e-5, help="learning rate of verifier training")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    epochs = args.epoch
    batch_size = args.batch
    lr = args.lr
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

        # extract data on validation set to train the verifier

        texts = []
        labels = [] # if the question is answered correctly, 1; else 0

        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name)
            if lm_utils.answer_parsing(response) == d["answer"]:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(original_prompt + "\n" + response)
        
        # train the verifier

        lm_utils.mlm_text_classifier(texts, labels, epochs, batch_size, lr)
        
        # obtain abstain flags for test set

        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name)
            label, prob = lm_utils.text_classifier_inference(original_prompt + "\n" + response)
            # print(label, prob)
            if label == 1:
                abstain_flags.append(0)
                abstain_scores.append(1-prob)
            else:
                abstain_flags.append(1)
                abstain_scores.append(prob)

    print("------------------")
    print("Approach: verifier")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")