import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
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

        # set a threshold over token probability for abstain

        correct_labels_dev = []
        option_to_ind = {"A": 0, "B": 1, "C": 2, "D": 3}
        probs = []
        target = []
        for d in tqdm(data["dev"]):
            correct_answer = d["answer"]
            target.append(option_to_ind[correct_answer])
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response, token_probs = lm_utils.llm_response(original_prompt, model_name, probs=True)
            # print(response, token_probs)
            # print("------------------")
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_labels_dev.append(1)
            else:
                correct_labels_dev.append(0)
            prob_max = None
            chosen_option = None
            try:
                for token in token_probs.keys():
                    if token.strip() in option_to_ind.keys():
                        prob_max = token_probs[token]
                        chosen_option = token.strip()
                        break
            except:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            if chosen_option == None:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            prob_distribution = [0, 0, 0, 0]
            prob_distribution[option_to_ind[chosen_option]] = prob_max
            # evenly split between other options, since we only care about the most likely option / the one generated in greedy decoding
            for i in range(4):
                if i != option_to_ind[chosen_option]:
                    prob_distribution[i] = (1 - prob_max) / 3
            probs.append(prob_distribution)

        # determine optimal threshold for abstain

        prob_maximum = max([max(prob) for prob in probs])
        prob_minimum = min([max(prob) for prob in probs])

        min_error = 1e6
        best_threshold = 0
        for threshold in range(1, 100):

            # no 100% abstain or 100% answer
            if threshold / 100.0 >= prob_maximum or threshold / 100.0 <= prob_minimum:
                continue

            error = 0
            for i in range(len(correct_labels_dev)):
                if max(probs[i]) < float(threshold/100.0): # probs lower than threshold -> should abstain
                    if correct_labels_dev[i] == 1: # but correct, error
                        error += 1
                else: # probs higher than threshold -> should answer
                    if correct_labels_dev[i] == 0: # but incorrect, error
                        error += 1
            if error < min_error:
                min_error = error
                best_threshold = float(threshold/100.0)
                # print("best threshold:", best_threshold)
                # print("best error:", min_error)

        # obtain abstain flags for test set

        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response, token_probs = lm_utils.llm_response(original_prompt, model_name, probs=True)
            prob_max = None
            chosen_option = None
            try:
                for token in token_probs.keys():
                    if token.strip() in option_to_ind.keys():
                        prob_max = token_probs[token]
                        chosen_option = token.strip()
                        break
            except:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            if chosen_option == None:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B", "C", "D"])
                prob_max = 0.25
            prob_distribution = [0, 0, 0, 0]
            prob_distribution[option_to_ind[chosen_option]] = prob_max
            # evenly split between other options, since we only care about the most likely option / the one generated in greedy decoding
            for i in range(4):
                if i != option_to_ind[chosen_option]:
                    prob_distribution[i] = (1 - prob_max) / 3

            if prob_distribution[option_to_ind[chosen_option]] < best_threshold:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)
            abstain_scores.append(1-prob_distribution[option_to_ind[chosen_option]]) # abstain likelihood

    # print(correct_flags)
    # print(abstain_flags)
    # print(abstain_scores)

    print("------------------")
    print("Approach: probability")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")