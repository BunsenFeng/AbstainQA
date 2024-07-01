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

            options = []
            for key in d["choices"].keys():
                options.append(d["choices"][key])
            options.append("None of the above")
            # shuffle options
            random.shuffle(options)
            problem = {}
            symbols = ["A", "B", "C", "D", "E"]
            for i in range(len(options)):
                problem[symbols[i]] = options[i]
            # find out which is none of the above
            nota_answer = ""
            for key in problem.keys():
                if problem[key] == "None of the above":
                    nota_answer = key
            prompt = "Question: " + d["question"] + "\n"
            for key in problem.keys():
                prompt += (key + ": " + problem[key] + "\n")
            prompt += "Choose one answer from the above choices. The answer is"
            response, probs = lm_utils.llm_response(prompt, model_name, probs=True)
            if lm_utils.answer_parsing(response) == nota_answer:
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)

            try:
                for key in probs.keys():
                    found = False
                    for symbol in symbols:
                        if key.strip().lower() == symbol.lower():
                            if symbol == nota_answer:
                                abstain_scores.append(probs[key])
                                found = True
                                break
                            else:
                                abstain_scores.append(1 - probs[key])
                                found = True
                                break
                    if found:
                        break
            except:
                abstain_scores.append(0.5)
    
    print("------------------")
    print("Approach: NOTA")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")