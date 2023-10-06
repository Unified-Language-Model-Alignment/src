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
            # "output": datasets.Sequence(datasets.Value("string")),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
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

                    assist_idx = chosen.rfind("\n\nAssistant: ")
                    r_accept = chosen[assist_idx+13:].strip()

                    yield key, {
                        "instruction": chosen[:assist_idx+13].strip(),
                        "output": r_accept,
                        "history": []
                    }
                    key += 1
