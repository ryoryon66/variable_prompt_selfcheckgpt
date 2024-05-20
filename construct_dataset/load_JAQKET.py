# load jsonl data
import jsonlines


def load_jaqket(path: str = "dev2_questions.json", n_samples=1000):
    res = {}
    with open(path, "r") as f:
        reader = jsonlines.Reader(f)

        res = [x for x in reader]

    res = res[:n_samples]

    # reduce key to "qid" and "question" and "answer"

    res = [
        {"qid": x["qid"], "question": x["question"], "answer": x["answer_entity"]}
        for x in res
    ]

    return res


if __name__ == "__main__":
    d = load_jaqket("dev2_questions.json", n_samples=10)
    from pprint import pprint

    pprint(d)
