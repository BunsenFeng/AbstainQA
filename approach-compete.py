import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-a", "--another_model", help="which model to use for conflicting knowledge generation and challenging")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")

    args = argParser.parse_args()
    model_name = args.model
    another_model = args.another_model
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

        original_answer = []
        for d in tqdm(data["test"]):
            # original answer correct flag
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False)
            original_answer.append(lm_utils.answer_parsing(response))
            # print(response)
            # print(lm_utils.answer_parsing(response))
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

        alternative_knowledge = []
        lm_utils.wipe_model()
        lm_utils.llm_init(another_model)
        for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
            # generate a conflicting knowledge passage
            prompt_generate_conflict = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                prompt_generate_conflict += (key + ": " + d["choices"][key] + "\n")
            remaining_choices = list(d["choices"].keys())
            try:
                remaining_choices.remove(original_answer[i])
            except:
                pass
            random.shuffle(remaining_choices)
            wrong_answer = remaining_choices[0]
            prompt_generate_conflict += "Generate a knowledge paragraph about " + wrong_answer + "."
            response = lm_utils.llm_response(prompt_generate_conflict, model_name, probs=False, temperature=1)
            alternative_knowledge.append(response)
            # print("--------------------")
            # print(prompt_generate_conflict)
            # print(response)
            # print(wrong_answer)
            # print(d["answer"])
            # print("--------------------")

        assert len(alternative_knowledge) == len(data["test"])

        lm_utils.wipe_model()
        lm_utils.llm_init(model_name)
        for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
            # the orginal model answers when presented with conflicting info
            conflict_prompt = "Answer the question with the following knowledge: feel free to ignore irrelevant or wrong information.\n\nKnowledge: " + alternative_knowledge[i] + "\n"
            conflict_prompt += "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                conflict_prompt += (key + ": " + d["choices"][key] + "\n")
            conflict_prompt += "Choose one answer from the above choices. The answer is"
            response, probs = lm_utils.llm_response(conflict_prompt, model_name, probs=True, temperature=1)

            if lm_utils.answer_parsing(response) == original_answer[i]:
                abstain_flags.append(0)
                if original_answer[i] in probs.keys():
                    abstain_scores.append(1-probs[original_answer[i]])
                elif " " + original_answer[i] in probs.keys():
                    abstain_scores.append(1-probs[" " + original_answer[i]])
            else:
                abstain_flags.append(1)
                if lm_utils.answer_parsing(response) in probs.keys():
                    abstain_scores.append(probs[lm_utils.answer_parsing(response)])
                elif " " + lm_utils.answer_parsing(response) in probs.keys():
                    abstain_scores.append(probs[" " + lm_utils.answer_parsing(response)])
    
    assert len(correct_flags) == len(abstain_flags)
    print("------------------")
    print("Approach: compete")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")