import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input", help="sft data name, e.g. mmlu-mistral-instruction-tuning")
    argParser.add_argument("-m", "--model", help="model name, e.g. mistral")
    argParser.add_argument("-p", "--parent_directory", default="./sft_data/", help="parent directory") # other_checkpoint/
    argParser.add_argument("-e", "--epochs", default=5, help="number of epochs")

    args = argParser.parse_args()
    input = args.input
    model_name = args.model
    parent_directory = args.parent_directory
    epochs = int(args.epochs)

    dataset = load_dataset("json", data_files="sft_data/" + input + ".jsonl", split="train")
    # print(len(dataset))

    if model_name == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif model_name == "llama2_7b":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "llama2_13b":
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == "llama2_70b":
        model_name = "meta-llama/Llama-2-70b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    peft_config = LoraConfig(
        r=64,  # the rank of the LoRA matrices
        lora_alpha=16, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
        modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    )

    training_args = SFTConfig(
        output_dir= parent_directory + input,  # the directory where the model will be saved
        # report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.1,
        gradient_checkpointing=True,
        # eval_strategy="epoch",
        num_train_epochs=epochs,
        # logging strategies 
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        max_seq_length=1024,
        packing=True,
        run_name=input,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config = peft_config,
    )

    trainer.train()
    trainer.save_model(parent_directory + input)