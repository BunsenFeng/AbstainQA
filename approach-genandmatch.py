import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm

def rule_matching(answer_str, option):
    answer_str = answer_str.lower()
    option = option.lower()
    for i in range(len(option)):
        if option[i] > 'z' or option[i] < 'a':
            option = option.replace(option[i], " ")
    option = option.split(" ")
    for word in option:
        if len(word) == 1:
            continue
        if not word in answer_str:
            return False
    return True

# answer_str = "The answer is Donald Trump."
# option = "Donald J. Trump"
# print(rule_matching(answer_str, option))

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
    abstain_flags_self = [] # rule-based matching
    abstain_scores_self = None
    abstain_flags_lm = [] # lm matching
    abstain_scores_lm = []

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

            original_prompt = "Question: " + d["question"] + "\nAnswer:" # no option provided
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)

            # rule-based matching
            matched_flag = False
            for key in d["choices"].keys():
                if rule_matching(response, d["choices"][key]):
                    matched_flag = True
                    abstain_flags_self.append(0)
                    break
            if not matched_flag:
                abstain_flags_self.append(1)

            # LM matching
            prompt = "Question: " + d["question"] + "\nProposed Answer: " + response + "\nOptions:"
            for key in d["choices"].keys():
                prompt += (" " + key + ". " + d["choices"][key])
            prompt += "\nDoes the proposed answer exist in the options?"
            response, probs = lm_utils.llm_response(prompt, model_name, probs=True)
            probs_lower = {}
            for key in probs.keys():
                probs_lower[key.lower()] = probs[key]
            if "yes" in response.lower():
                abstain_flags_lm.append(0)
                if "yes" in probs_lower.keys():
                    abstain_scores_lm.append(probs_lower["yes"])
                elif " yes" in probs_lower.keys():
                    abstain_scores_lm.append(probs_lower[" yes"])
                else:
                    abstain_scores_lm.append(0.5)
            elif "no" in response.lower():
                abstain_flags_lm.append(1)
                if "no" in probs_lower.keys():
                    abstain_scores_lm.append(probs_lower["no"])
                elif " no" in probs_lower.keys():
                    abstain_scores_lm.append(probs_lower[" no"])
                else:
                    abstain_scores_lm.append(0.5)
            else:
                print("Error: neither yes nor no in response")
                abstain_flags_lm.append(random.randint(0, 1))
                abstain_scores_lm.append(0.5)
    
    assert len(correct_flags) == len(abstain_flags_self)
    assert len(correct_flags) == len(abstain_flags_lm)
    assert len(correct_flags) == len(abstain_scores_lm)

    print("------------------")
    print("Approach: GenandMatch")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Setting:", "rule matching")
    print(metrics.compute_metrics(correct_flags, abstain_flags_self, abstain_scores_self))
    print("------------------")

    print("------------------")
    print("Approach: GenandMatch")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Setting:", "lm matching")
    print(metrics.compute_metrics(correct_flags, abstain_flags_lm, abstain_scores_lm))
    print("------------------")