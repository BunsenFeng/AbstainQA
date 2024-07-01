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
    argParser.add_argument("-t", "--type", help="approach type, self or others") # "self", "others"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    approach_type = args.type
    portion = args.portion

    lm_utils.llm_init(model_name)

    correct_flags = []
    abstain_flags = []
    abstain_scores = []

    with open("data/" + dataset + ".json", "r") as f:

        data = json.load(f)

        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        answers = []
        feedback_1 = []
        feedback_2 = []
        feedback_3 = []
        
        # obtain correct flags
            
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
            answers.append(response)

        # obtain feedbacks

        prompt_feedback_list = []

        for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer is"

            prompt_feedback = original_prompt + " " + answers[i].strip() + "\nPlease review the proposed answer and provide feedback on its correctness.\nFeedback:"
            prompt_feedback_list.append(prompt_feedback)
        
        if approach_type == "self": # expert reviewers with self-specialized domains
            for prompt_feedback in tqdm(prompt_feedback_list):
                # generate knowledge from different expert's perspectives: facts, multi-hop, and commonsense
                prompt_feedback_experts = []
                for domain_name in ["factual information", "multi-hop reasoning", "commonsense knowledge"]:
                    expert_prompt = "Generate some knowledge about the question, focusing on " + domain_name + ".\n" + prompt_feedback.split("\n")[0] + "\nKnowledge:"
                    prompt_feedback_experts.append("Knowledge: " + lm_utils.llm_response(expert_prompt, model_name, probs=False, temperature=1).split("\n")[0].strip() + "\n" + prompt_feedback)

                assert len(prompt_feedback_experts) == 3

                response = lm_utils.llm_response(prompt_feedback_experts[0], model_name, probs=False, temperature=1)
                feedback_1.append(response.split("\n")[0].strip())
                response = lm_utils.llm_response(prompt_feedback_experts[1], model_name, probs=False, temperature=1)
                feedback_2.append(response.split("\n")[0].strip())
                response = lm_utils.llm_response(prompt_feedback_experts[2], model_name, probs=False, temperature=1)
                feedback_3.append(response.split("\n")[0].strip())
        elif approach_type == "others":
            models_list = ["mistral", "llama2_70b", "chatgpt"]
            # current model
            for prompt_feedback in tqdm(prompt_feedback_list):
                response = lm_utils.llm_response(prompt_feedback, model_name, probs=False, temperature=1)
                feedback_1.append(response)
            models_list.remove(model_name)
            # other models
            model_now = models_list[0]
            lm_utils.wipe_model()
            lm_utils.llm_init(model_now)
            for prompt_feedback in tqdm(prompt_feedback_list):
                response = lm_utils.llm_response(prompt_feedback, model_now, probs=False, temperature=1)
                feedback_2.append(response)
            model_now = models_list[1]
            lm_utils.wipe_model()
            lm_utils.llm_init(model_now)
            for prompt_feedback in tqdm(prompt_feedback_list):
                response = lm_utils.llm_response(prompt_feedback, model_now, probs=False, temperature=1)
                feedback_3.append(response)
        
        # obtain abstain flags and scores

        prompt_area_chair_list = []
        assert len(data["test"]) == len(answers) == len(feedback_1) == len(feedback_2) == len(feedback_3)
        for i in range(len(data["test"])):
            d = data["test"][i]
            prompt_area_chair = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                prompt_area_chair += (key + ": " + d["choices"][key] + "\n")
            prompt_area_chair += "Choose one answer from the above choices. The answer is " + answers[i].strip() + "\n\nFeedback 1: " + feedback_1[i].strip() + "\n\nFeedback 2: " + feedback_2[i].strip() + "\n\nFeedback 3: " + feedback_3[i].strip() + "\n\nBased on the feedback, the proposed answer is:\nA. True\nB. False\nThe answer is"
            prompt_area_chair_list.append(prompt_area_chair)
            # print(prompt_area_chair)
            # print("--------------------")
        assert len(prompt_area_chair_list) == len(data["test"])
            
        if approach_type == "self":
            for prompt_area_chair in tqdm(prompt_area_chair_list):
                response, probs = lm_utils.llm_response(prompt_area_chair, model_name, probs=True)

                if lm_utils.answer_parsing(response) == "A":
                    abstain_flags.append(0)
                elif lm_utils.answer_parsing(response) == "B":
                    abstain_flags.append(1)
                else:
                    print("Error: abstain flag not found")
                    abstain_flags.append(random.randint(0, 1))
                try:
                    if abstain_flags[-1] == 0:
                        if "A" in probs.keys():
                            abstain_scores.append(probs["A"])
                        elif " A" in probs.keys():
                            abstain_scores.append(probs[" A"])
                    else:
                        if "B" in probs.keys():
                            abstain_scores.append(probs["B"])
                        elif " B" in probs.keys():
                            abstain_scores.append(probs[" B"])
                except:
                    abstain_scores.append(0.5)
        elif approach_type == "others":
            lm_utils.wipe_model()
            lm_utils.llm_init("chatgpt") # always area-charing with chatgpt
            for prompt_area_chair in tqdm(prompt_area_chair_list):
                response, probs = lm_utils.llm_response(prompt_area_chair, "chatgpt", probs=True)

                if lm_utils.answer_parsing(response) == "A":
                    abstain_flags.append(0)
                elif lm_utils.answer_parsing(response) == "B":
                    abstain_flags.append(1)
                else:
                    print("Error: abstain flag not found")
                    abstain_flags.append(random.randint(0, 1))
                # print(probs)
                try:
                    if abstain_flags[-1] == 0:
                        if "A" in probs.keys():
                            abstain_scores.append(probs["A"])
                        elif " A" in probs.keys():
                            abstain_scores.append(probs[" A"])
                    else:
                        if "B" in probs.keys():
                            abstain_scores.append(probs["B"])
                        elif " B" in probs.keys():
                            abstain_scores.append(probs[" B"])
                except:
                    abstain_scores.append(0.5)
    # print(abstain_scores)
        
    print("------------------")
    print("Approach: Cooperate")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Type:", approach_type)
    print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
    print("------------------")