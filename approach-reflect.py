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
        
            response += "\nThe above answer is:\nA. True\nB. False\nThe answer is"
            response, probs = lm_utils.llm_response(response, model_name, probs=True)
            # print(response)
            if lm_utils.answer_parsing(response) == "A":
                abstain_flags.append(0)
            else:
                abstain_flags.append(1)

            option_to_ind = {"a": 0, "b": 1, "true": 0, "false": 1}

            try:
                for token in probs.keys():
                    if token.strip().lower() in option_to_ind.keys():
                        if option_to_ind[token.strip().lower()] == 0:
                            abstain_scores.append(1 - probs[token])
                            break
                        else:
                            abstain_scores.append(probs[token])
                            break
            except:
                print("option probs failed, uniform assignment")
                chosen_option = random.choice(["A", "B"])
                abstain_scores.append(0.5)

    print("------------------")
    print("Approach: reflect")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")