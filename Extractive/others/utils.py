import re

import rouge

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/3/l): {:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        results_dict["rouge_l_precision"] * 100
    )


def calculate_rouge(can_path, gold_path):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=False,
                            apply_avg=True,
                            alpha=0.5,
                            stemming=True)
    with open(can_path, 'r') as f:
        all_hypothesis = []
        pred = f.read()
        pred = pred.split('\n')
        for h in pred:
            h = h.replace('<q> ', '\n')
            all_hypothesis.append(h)
    with open(gold_path, 'r') as f:
        all_references = []
        gold = f.read()
        gold = gold.split('\n')
        for g in gold:
            g = g.replace('<q>', '\n')
            all_references.append(g)
    scores = evaluator.get_scores(all_hypothesis, all_references)
    results_dict = {}
    results_dict["rouge_1_f_score"] = scores["rouge-1"]['f']
    results_dict["rouge_1_recall"] = scores["rouge-1"]['r']
    results_dict["rouge_1_precision"] = scores["rouge-1"]['p']
    results_dict["rouge_2_f_score"] = scores["rouge-2"]['f']
    results_dict["rouge_2_recall"] = scores["rouge-2"]['r']
    results_dict["rouge_2_precision"] = scores["rouge-2"]['p']
    results_dict["rouge_l_f_score"] = scores["rouge-l"]['f']
    results_dict["rouge_l_recall"] = scores["rouge-l"]['r']
    results_dict["rouge_l_precision"] = scores["rouge-l"]['p']
    return results_dict
