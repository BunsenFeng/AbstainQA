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

        # get correct_flags
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
        
            # TODO: implement your method to obtain abstain_flags and abstain_scores
            abstain_flags.append(random.randint(0, 1)) # 0 means no abstain and answer, 1 means abstain and no answer
            abstain_scores.append(random.random()) # 1 means absolutely abstain, 0 means absolutely not, 0.5 means half-half

    print("------------------")
    print("Approach: your_approach")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")