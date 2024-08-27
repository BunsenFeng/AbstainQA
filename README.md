# AbstainQA Repository

This is the official repo for [Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration](https://arxiv.org/abs/2402.00367) @ ACL 2024.

### Environment

```
conda env create -f abstainqa.yaml
conda activate abstainqa
export OPENAI_API_KEY="YOUR_KEY"
```

### Methods

We provide the implementation of 13 baselines and proposed approaches in the paper. Each `approach-<name>.py` file contains the implementation of the corresponding approach. Shared parameters for each approach:

```
-m MODEL, --model MODEL
                        which language model to use: "mistral", "llama2_7/13/70b", "chatgpt"
-d DATASET, --dataset DATASET
                        which dataset in data/: "mmlu", "knowledge_crosswords", "hellaswag", "propaganda", "ambigqa", "electionqa23"
-o PORTION, --portion PORTION
                        portion of the dataset to use, 0-1
```

These are the default models and datasets we provide in the implementation: more on adding your own later. Portion (0-1) means only evaluating on the first `x%` of the dataset in case the LLM is large and evaluation is slow. We introduce the methods in the following:

#### Calibration: `approach-probability.py`

The `Token Probability` approach in Section 2.1.

```
approach-probability.py [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Calibration: `approach-temperature.py`

The `Temperature Scaling` approach in Section 2.1.

```
approach-temperature.py [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Calibration: `approach-askcalibrate.py`

The `Ask for Calibration` approach in Section 2.1.

```
approach-askcalibrate.py [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Training: `approach-embedding.py`

The `Hidden Layers` approach in Section 2.2. Not compatible with black-box models.

```
approach-embedding.py [-m MODEL] [-d DATASET] [-o PORTION] [-p PHASE]

options:
  -p PHASE, --phase PHASE
                        one or two: "one" for evaluating on validation and test sets, "two" for extracting embeddings, linear probing, and obtain abstain flags
```

Please run `-p one` first to evaluate and save predictions, then `-p two` to extract embeddings, train a linear model, and obtain abstain decisions.

#### Training: `approach-verifier.py`

The `External Verifier` approach in Section 2.2.

```
approach-verifier.py [-m MODEL] [-d DATASET] [-o PORTION] [-e EPOCH] [-b BATCH] [-l LR]

options:
  -e EPOCH, --epoch EPOCH
                        epochs of verifier training
  -b BATCH, --batch BATCH
                        batch size of verifier training
  -l LR, --lr LR        learning rate of verifier training
```

Default hyperparameters are provided for these options of verifier training, so you don't need to change them unless you want to.

#### Training: `approach-instructiontune.py`

The `Instruction Tuning` approach in Section 2.2.

```
approach-instructiontune.py [-m MODEL] [-d DATASET] [-o PORTION] [-s SETTING] [-t TUNED_MODEL_NAME]

options:
  -s SETTING, --setting SETTING
                        generate or evaluate
  -t TUNED_MODEL_NAME, --tuned_model_name TUNED_MODEL_NAME
                        name of the tuned model, either chatgpt via OpenAI API or local/hf copy of tuned model path
```

1) Run `-s generate` first to generate SFT data for abstention.
2) SFT the model. If `chatgpt`, do it on your own with the OpenAI API. If other open models, we provide `sft.py` as a bare-bone implementation. Feel free to use your own SFT code though.
3) Run `-s evaluate` with `-t <tuned_model>`, OpenAI model ID for chatgpt and local/hf path for open models.

```
sft.py [-i INPUT] [-m MODEL] [-p PARENT_DIRECTORY] [-e EPOCHS]

options:
  -i INPUT, --input INPUT
                        sft data name, e.g. mmlu-mistral-instruction-tuning
  -m MODEL, --model MODEL
                        model name, e.g. mistral
  -p PARENT_DIRECTORY, --parent_directory PARENT_DIRECTORY
                        parent directory, default sft-data/
  -e EPOCHS, --epochs EPOCHS
                        number of epochs, default 5
```

#### Prompting: `approach-reflect.py`

The `Self-Reflect` approach in Section 2.3.

```
approach-reflect.py [-h] [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Prompting: `approach-moreinfo.py`

The `More Information` approach in Section 2.3.

```
approach-moreinfo.py [-h] [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Prompting: `approach-genandmatch.py`

The `Generate and Match` approach in Section 2.3.

```
approach-genandmatch.py [-h] [-m MODEL] [-d DATASET] [-o PORTION]
```

Note that two results will come out at once: one with LM-based answer choice matching and one with rule-based matching.

#### Consistency: `approach-nota.py`

The `None-of-the-Above` approach in Section 2.4.

```
approach-nota.py [-h] [-m MODEL] [-d DATASET] [-o PORTION]
```

#### Consistency: `approach-scthreshold.py`

The `Self-Consistency Threshold` approach in Section 2.4.

```
approach-scthreshold.py [-h] [-m MODEL] [-d DATASET] [-o PORTION] [-p PATH]

options:
  -p PATH, --path PATH  number of paths to use for self consistency, default: 5
```

`-p` governs how many CoT paths to generate and the base of the plurality. Increasing this will lead to much more inference time, because the implementation now is not batch-inference: this is to be compatible with other real-time exchange methods.

#### Collaboration: `approach-cooperate.py`

The `Cooperate` approach in Section 2.5.

```
approach-cooperate.py [-h] [-m MODEL] [-d DATASET] [-o PORTION] [-t TYPE]

options:
  -t TYPE, --type TYPE  approach type, self or others
```

`-t self` means self-feedbacks by specializing the LLM itself into different expert domains (math, facts, etc.) for feedback generation and make abstain decisions. `-t others` incidates having other LLMs generate feedback on the proposed answer. By default we have `mistral`, `llama2_70b`, and `chatgpt`: the one you chose with `-m` will propose an answer, and these three models will generate feedbacks, finally `chatgpt` makes an abstain decision (like an area chair) with the proposed answer and feedback from other LLMs. Change the models in line 86 if you want: these will have to be implemented in `lm_utils.py`.

#### Collaboration: `approach-compete.py`

The `Compete` approach in Section 2.5.

```
approach-compete.py [-h] [-m MODEL] [-a ANOTHER_MODEL] [-d DATASET] [-o PORTION]

options:
  -a ANOTHER_MODEL, --another_model ANOTHER_MODEL
                        which model to use for conflicting knowledge generation and challenging
```

Please specify a model in `-a`, same or different from `-m`, to challenge the `-m` and generate conflicting knowledge paragraphs.

#### Your Approach

`approach-yours.py` provides a skeleton for adding your approach. Basically, just adding ways of getting `abstain_flags` and `abstain_scores` (if any) indicating whether the LLM should abstain for questions based on your methodology.

### Models

`lm_utils.py` provides inference code for `mistral`, `llama2_7b`, `llama2_13b`, `llama2_70b`, and `chatgpt`. If you want to add new models, add it in both `lm_init()` where you initialize the model and tokenizer; and `llm_response()` where you generate text with it and provide token probabilities (if any).

### Datasets

We provide datasets in `data/` for `mmlu`, `knowledge_crosswords`, `hellaswag`, `propaganda`, `ambigqa`, and `electionqa23`. If you want to add new datasets, add it in `data/` and follow the same format as the existing ones. These datasets are multiple-choice QA datasets, while we plan to support non-MC datasets in future work. Please check out the paper for references to these datasets.

### Metrics

`metrics.py` provides the implementation of AbstainQA metrics (Section 3) calcualted from `correct_flags`, `abstain_flags`, and `abstain_scores` (if any). Feel free to add your AbstainQA metric and add it to the return dictionary.

### Citation

```
@inproceedings{feng-etal-2024-dont,
    title = "Don{'}t Hallucinate, Abstain: Identifying {LLM} Knowledge Gaps via Multi-{LLM} Collaboration",
    author = "Feng, Shangbin  and
      Shi, Weijia  and
      Wang, Yike  and
      Ding, Wenxuan  and
      Balachandran, Vidhisha  and
      Tsvetkov, Yulia",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.786",
    pages = "14664--14690",
    abstract = "Despite efforts to expand the knowledge of large language models (LLMs), knowledge gaps{---}missing or outdated information in LLMs{---}might always persist given the evolving nature of knowledge. In this work, we study approaches to identify LLM knowledge gaps and abstain from answering questions when knowledge gaps are present. We first adapt existing approaches to model calibration or adaptation through fine-tuning/prompting and analyze their ability to abstain from generating low-confidence outputs. Motivated by their failures in self-reflection and over-reliance on held-out sets, we propose two novel approaches that are based on model collaboration, i.e., LLMs probing other LLMs for knowledge gaps, either cooperatively or competitively. Extensive experiments with three LLMs on four QA tasks featuring diverse knowledge domains demonstrate that both cooperative and competitive approaches to unveiling LLM knowledge gaps achieve up to 19.3{\%} improvements on abstain accuracy against the strongest baseline. Further analysis reveals that our abstention methods pinpoint failure cases in retrieval augmentation and knowledge gaps in multi-hop reasoning.",
}
```

ACL bibtex coming soon. PRs are welcome for any issues or improvements. Enjoy AbstainQA!
