import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import argparse

sys.path.append("../")

import utils.llm
import utils.scoring

parser = argparse.ArgumentParser(description="Run detection")


parser.add_argument("--annotated", type=str, help="path to the annotation csv")
parser.add_argument("--samples", type=str, help="path to the samples")
parser.add_argument(
    "--num_samples", type=int, default=30, help="number of samples to use"
)
parser.add_argument(
    "--output", type=str, default="score_result.csv", help="result output path"
)


args = parser.parse_args()

ANNOTATED_PATH = args.annotated
DATA_SAMPLES_PATH = args.samples
NUM_SAMPLES = args.num_samples
OUTPUT_PATH = args.output


annotated_df = pd.read_csv(ANNOTATED_PATH)
sampled_df = pd.read_csv(DATA_SAMPLES_PATH)


annotated_df = annotated_df[:200]
num_questions = len(annotated_df)
score_df = pd.DataFrame(columns=["query", "score_pr", "score_re", "score_f1"])

for i in tqdm(range(num_questions)):
    # logger.info("processing {} / {}".format(i,num_questions))

    qid = annotated_df.iloc[i]["qid"]
    response: str = annotated_df.iloc[i]["generated_answer"]
    samples = list(sampled_df[sampled_df["qid"] == qid]["sample"])[:NUM_SAMPLES]

    pr_scores, re_scores, f1_scores = utils.scoring.get_hallucination_score(
        response, samples
    )

    # score_df = score_df.append({
    #         'qid':qid,
    #         'pr_score':",".join([str(x) for x in pr_scores]),
    #         're_score':",".join([str(x) for x in re_scores]),
    #         'f1_score':",".join([str(x) for x in f1_scores]),
    #     },
    #     ignore_index=True)

    # use concat instead of append

    score_df = pd.concat(
        [
            score_df,
            pd.DataFrame(
                {
                    "qid": qid,
                    "pr_score": ",".join([str(x) for x in pr_scores]),
                    "re_score": ",".join([str(x) for x in re_scores]),
                    "f1_score": ",".join([str(x) for x in f1_scores]),
                },
                index=[0],
            ),
        ]
    )


# logger.info("done")

result_df = pd.merge(annotated_df, score_df, on="qid")
result_df.to_csv(OUTPUT_PATH, index=False)

result_df = pd.read_csv(OUTPUT_PATH)


def extract_label_and_scores():
    df = result_df.copy()

    df = df[["qid", "factuality", "pr_score", "re_score", "f1_score", "question"]]

    label_and_scores = {"label": [], "pr": [], "re": [], "f1": []}

    # set data type
    df["factuality"] = df["factuality"].astype(str)

    for i in range(df.shape[0]):
        labels = df.iloc[i]["factuality"].split(",")
        pr_scores = df.iloc[i]["pr_score"].split(",")
        re_scores = df.iloc[i]["re_score"].split(",")
        f1_scores = df.iloc[i]["f1_score"].split(",")

        labels = [1 - int(x) for x in labels]
        pr_scores = [float(x) for x in pr_scores]
        re_scores = [float(x) for x in re_scores]
        f1_scores = [float(x) for x in f1_scores]

        assert len(labels) == len(
            pr_scores
        ), f"labels: {labels}, pr_scores: {pr_scores},i: {i},q: {df.iloc[i]['question']}"

        label_and_scores["label"].extend(labels)
        label_and_scores["pr"].extend(pr_scores)
        label_and_scores["re"].extend(re_scores)
        label_and_scores["f1"].extend(f1_scores)

    label_and_scores["label"] = np.array(label_and_scores["label"])
    label_and_scores["pr"] = np.array(label_and_scores["pr"])
    label_and_scores["re"] = np.array(label_and_scores["re"])
    label_and_scores["f1"] = np.array(label_and_scores["f1"])

    return label_and_scores


label_and_scores = extract_label_and_scores()


roc_aucs = {
    "pr": roc_auc_score(label_and_scores["label"], label_and_scores["pr"]),
    "re": roc_auc_score(label_and_scores["label"], label_and_scores["re"]),
    "f1": roc_auc_score(label_and_scores["label"], label_and_scores["f1"]),
    "ave": roc_auc_score(
        label_and_scores["label"],
        (label_and_scores["pr"] + label_and_scores["re"] + label_and_scores["f1"]) / 3,
    ),
    "random": roc_auc_score(
        label_and_scores["label"], np.random.rand(len(label_and_scores["label"]))
    ),
}

print(f"the auc-roc is {roc_aucs}")
