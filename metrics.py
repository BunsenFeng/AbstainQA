import numpy as np

def compute_metrics(correct_flags, abstain_flags, abstain_scores = None):
    # correct_flags: a list of [0,1]s representing the correctness of each QA answered by the LLM
    # abstain_flags: a list of [0,1]s representing whether the LLM abstained from answering each QA
    # abstain_scores: a list of floats from 0 to 1 representing the confidence of the LLM in abstaining
    # returns: a dictionary of metrics

    assert len(correct_flags) == len(abstain_flags)

    # group A: answered and correct
    # group B: abstained and correct
    # group C: answered and incorrect
    # group D: abstained and incorrect
    A = 0
    B = 0
    C = 0
    D = 0
    for i in range(len(correct_flags)):
        if abstain_flags[i]:
            if correct_flags[i]:
                B += 1
            else:
                D += 1
        else:
            if correct_flags[i]:
                A += 1
            else:
                C += 1
        
    # reliable accuracy: accuracy of the LLM on the questions it answered
    try:
        reliable_accuracy = A / (A + C)
    except:
        reliable_accuracy = None

    # effective reliability: correct 1, incorrect -1, abstained 0
    effective_reliability = (A - C) / (A + B + C + D)

    # abstain accuracy: accuracy of the LLM abstain decisions, how many times correct_flags == !abstain flags
    abstain_accuracy = (A + D) / (A + B + C + D)

    # abstain precision: how many abstains is right among all abstains
    try:
        abstain_precision = D / (B + D)
    except:
        abstain_precision = None

    # abstain recall: how many abstains is right among all incorrect answers
    try:
        abstain_recall = D / (C + D)
    except:
        abstain_recall = None

    # abstain ECE: bucket abstain confidence into 10 buckets (0:0.1:1), compute the expected calibration error
    if abstain_scores is not None and max(abstain_scores) != min(abstain_scores):

        # rescale abstain scores to 0-1 before calculation
        max_score = max(abstain_scores)
        min_score = min(abstain_scores)
        for i in range(len(abstain_scores)):
            abstain_scores[i] = (abstain_scores[i] - min_score) / (max_score - min_score)

        bucket_probs = [[] for i in range(10)]
        bucket_abstain = [[] for i in range(10)] # whether it should have abstained

        for i in range(len(abstain_scores)):
            if abstain_scores[i] == 1:
                bucket = 9
            else:
                bucket = int(abstain_scores[i] * 10)
            bucket_probs[bucket].append(abstain_scores[i])
            if correct_flags[i] == 1:
                bucket_abstain[bucket].append(0)
            else:
                bucket_abstain[bucket].append(1)
            
        bucket_ece = 0
        for i in range(10):
            if len(bucket_probs[i]) == 0:
                continue
            bucket_probs_avg = np.mean(bucket_probs[i])
            bucket_abstain_avg = np.mean(bucket_abstain[i])
            bucket_ece += abs(bucket_abstain_avg - bucket_probs_avg) * len(bucket_probs[i])
        bucket_ece /= len(abstain_scores)
    else:
        bucket_ece = None

    # abstain rate: what percentage of questions the LLM abstained from
    abstain_rate = (B + D) / (A + B + C + D)
            
    return {
        'reliable_accuracy': reliable_accuracy,
        'effective_reliability': effective_reliability,
        'abstain_accuracy': abstain_accuracy,
        'abstain_precision': abstain_precision,
        'abstain_recall': abstain_recall,
        'abstain_ece': bucket_ece,
        'abstain_rate': abstain_rate
    }

# correct_flags = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
# abstain_flags = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# abstain_scores = [0.1, 0.3, 0.3, 0.4, 0.4, 0.7, 0.7, 0.7, 0.7, 0.7]

# print(compute_metrics(correct_flags, abstain_flags, abstain_scores))