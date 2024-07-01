import os
import json
import torch
import argparse
import lm_utils
import metrics
import random
import time
import openai
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use: \"mistral\", \"llama2_7/13/70b\", \"chatgpt\"")
    argParser.add_argument("-d", "--dataset", help="which dataset in data/: \"mmlu\", \"knowledge_crosswords\", \"hellaswag\", \"propaganda\", \"ambigqa\", \"electionqa23\"")
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-s", "--setting", help="generate or evaluate") # "generate" first for generating instruction-tuning dataset, "evaluate" next if evaluating this approach with a tuned model
    argParser.add_argument("-t", "--tuned_model_name", default = None, help="name of the tuned model, either chatgpt via OpenAI API or local/hf copy of tuned model path") # tuned model name

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    setting = args.setting
    tuned_model_name = args.tuned_model_name
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

        if setting == "evaluate":
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

        # create instruction tuning dataset based on the dev set

        if setting == "generate":
            texts = []
            for d in tqdm(data["dev"]):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. The answer is"
                response = lm_utils.llm_response(original_prompt, model_name, probs=False)
                correct_flag = None
                if lm_utils.answer_parsing(response) == d["answer"]:
                    correct_flag = 1
                else:
                    correct_flag = 0

                if correct_flag:
                    texts.append({"messages": [{"role": "user", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.' " + original_prompt}, {"role": "assistant", "content": response}]})
                else:
                    texts.append({"messages": [{"role": "user", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.' " + original_prompt}, {"role": "assistant", "content": "Sorry, I don't have enough knowledge to answer this question."}]})

            # write texts in a jsonline format
            if not os.path.exists("sft_data"):
                os.makedirs("sft_data")
            with open("sft_data/" + dataset + "-" + model_name + "-instruction-tuning.jsonl", "w") as f:
                for text in texts:
                    f.write(json.dumps(text) + "\n")

        # getting abstain flags with the instruction-tuned version of ChatGPT

        if setting == "evaluate":
            if model_name == "chatgpt":
                assert tuned_model_name is not None
                for d in tqdm(data["test"]):
                    original_prompt = "Question: " + d["question"] + "\n"
                    for key in d["choices"].keys():
                        original_prompt += (key + ": " + d["choices"][key] + "\n")
                    original_prompt += "Choose one answer from the above choices. The answer is"

                    completion = openai.ChatCompletion.create(
                        model=tuned_model_name,
                        messages=[
                            {"role": "user", "content": "Answer the following question. If you don't have enough knowledge, abstain by saying 'sorry, I don't have enough knowledge to answer this question.' " + original_prompt}
                        ],
                        temperature = 0.1,
                        max_tokens=200,
                        # log_probs = 1,
                    )
                    time.sleep(0.1)
                    response = completion.choices[0].message["content"]
                    # print(response)
                    # print(lm_utils.answer_parsing(response))
                    if "sorry" in response.lower():
                        abstain_flags.append(1)
                    else:
                        abstain_flags.append(0)
            else:
                lm_utils.wipe_model()
                assert tuned_model_name is not None
                model = AutoModelForCausalLM.from_pretrained(tuned_model_name, device_map="auto", torch_dtype=torch.bfloat16)
                tokenizer = AutoTokenizer.from_pretrained(tuned_model_name)

                for d in tqdm(data["test"]):
                    original_prompt = "Question: " + d["question"] + "\n"
                    for key in d["choices"].keys():
                        original_prompt += (key + ": " + d["choices"][key] + "\n")
                    original_prompt += "Choose one answer from the above choices. The answer is"

                    input_ids = tokenizer(original_prompt, return_tensors="pt").input_ids.to("cuda")
                    outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, return_dict_in_generate=True, output_scores=True, temperature = 0.1, pad_token_id=tokenizer.eos_token_id)

                    input_length = input_ids.shape[1]
                    generated_ids = outputs.sequences[:, input_length:]
                    response = tokenizer.batch_decode(generated_ids)[0]

                    # print(lm_utils.answer_parsing(response))
                    if "sorry" in response.lower():
                        abstain_flags.append(1)
                    else:
                        abstain_flags.append(0)
        
        abstain_scores = None

    if setting == "evaluate":
        print("------------------")
        print("Approach: instructiontune")
        print("Model:", model_name)
        print("Dataset:", dataset)
        print(metrics.compute_metrics(correct_flags, abstain_flags, abstain_scores))
        print("------------------")