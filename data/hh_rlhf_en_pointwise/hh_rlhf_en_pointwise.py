import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "Anthropic Harmless dataset"
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/Anthropic/hh-rlhf"
_LICENSE = "mit"
_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
_URLS = {
    "train": [
        _URL + "harmless-base/train.jsonl.gz"
    ],
    "test": [
        _URL + "harmless-base/test.jsonl.gz"
    ]
}


class HhRlhfEn(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
            "score": datasets.Value("float32")
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": file_path["train"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": file_path["test"]
                }
            )
        ]

    # generate multi-turn chat for ChatGLM
    def _generate_examples(self, filepaths: List[str]) -> Dict[int, Dict[str, Any]]:
        key = 0
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    assist_idx = rejected.rfind("\n\nAssistant: ")
                    r_reject = rejected[assist_idx+13:].strip()
                    assist_idx = chosen.rfind("\n\nAssistant: ")
                    r_accept = chosen[assist_idx+13:].strip()

                    human_idx = chosen.rfind("\n\nHuman: ")
                    query = chosen[human_idx+9:assist_idx].strip()
                    prompt = chosen[:human_idx]
                    history = []

                    while prompt.rfind("\n\nAssistant: ") != -1:
                        assist_idx = prompt.rfind("\n\nAssistant: ")
                        human_idx = prompt.rfind("\n\nHuman: ")
                        if human_idx != -1:
                            old_query = prompt[human_idx+9:assist_idx].strip()
                            old_resp = prompt[assist_idx+13:].strip()
                            history.insert(0, (old_query, old_resp))
                        else:
                            break
                        prompt = prompt[:human_idx]

                    yield key, {
                        "instruction": query,
                        "output": r_accept,
                        "history": history,
                        "score": 1,
                    }
                    key += 1

                    yield key, {
                        "instruction": query,
                        "output": r_reject,
                        "history": history,
                        "score": 0,
                    }
                    key += 1
