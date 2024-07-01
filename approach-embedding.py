import os
import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import pipeline

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.out(self.activation(self.linear(x)))
    
def train_linear_model(model, train_data, train_labels, epochs=10, batch_size=8, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    return model

def test_linear_model(model, test_data):
    outputs = model(test_data)
    return torch.argmax(outputs, dim=1), outputs[0][0].item()

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-p", "--phase", help="one or two: \"one\" for evaluating on validation and test sets, \"two\" for extracting embeddings, linear probing, and obtain abstain flags")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    phase = args.phase
    portion = args.portion

    if phase == "one":
        lm_utils.llm_init(model_name)
    elif phase == "two":
        if model_name == "mistral":
            feature_extractor = pipeline("feature-extraction",framework="pt",model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
        elif model_name == "llama2_70b":
            feature_extractor = pipeline("feature-extraction",framework="pt",model="meta-llama/Llama-2-70b-chat-hf", device_map="auto", max_length=512, truncation=True)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []
    correct_flags_dev = []

    with open("data/" + dataset + ".json", "r") as f:

        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        if phase == "one":
            # correct flags for test set
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

            # correct flags for dev set
            for d in tqdm(data["dev"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = lm_utils.llm_response(original_prompt, model_name, probs=False)
                # print(response)
                # print(lm_utils.answer_parsing(response))
                if lm_utils.answer_parsing(response) == d["answer"]:
                    correct_flags_dev.append(1)
                else:
                    correct_flags_dev.append(0)

            # save it to a file

            if not os.path.exists("temp"):
                os.makedirs("temp")

            dict_to_save = {"correct_flags": correct_flags, "correct_flags_dev": correct_flags_dev}
            with open("temp/" + dataset + "_" + model_name + "_correct_flags.json", "w") as f:
                json.dump(dict_to_save, f)

        elif phase == "two":
            # load the saved file
            with open("temp/" + dataset + "_" + model_name + "_correct_flags.json", "r") as f:
                dict_to_load = json.load(f)
                correct_flags = dict_to_load["correct_flags"]
                correct_flags_dev = dict_to_load["correct_flags_dev"]
            
            # extract embeddings for dev set
            dev_embeddings = []
            for d in tqdm(data["dev"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = feature_extractor(original_prompt,return_tensors = "pt")[0].numpy().mean(axis=0)
                dev_embeddings.append(response)
            
            # extract embeddings for test set
            test_embeddings = []
            for d in tqdm(data["test"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = feature_extractor(original_prompt,return_tensors = "pt")[0].numpy().mean(axis=0)
                test_embeddings.append(response)

            # release gpu memory?
            lm_utils.wipe_model()
            
            # train linear model on dev set
            dev_embeddings = torch.tensor(dev_embeddings)

            linear_model = LinearModel(dev_embeddings.shape[1])
            linear_model = train_linear_model(linear_model, dev_embeddings, torch.tensor(correct_flags_dev))

            # obtain abstain flags for test set
            test_embeddings = torch.tensor(test_embeddings)

            for i in range(len(test_embeddings)):
                test_embedding = test_embeddings[i].unsqueeze(0)
                abstain_flag, abstain_score = test_linear_model(linear_model, test_embedding)
                abstain_flags.append(abstain_flag.item())
                abstain_scores.append(abstain_score)

    if phase == "two":   
        print("------------------")
        print("Approach: embedding")
        print("Model:", model_name)
        print("Dataset:", dataset)
        print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
        print("------------------")