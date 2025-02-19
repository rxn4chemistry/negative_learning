import json
from collections import defaultdict

from rxn_negative_learning.models.tokenization import BasicSmilesTokenizer
from rxn_negative_learning.utils.smiles_utils import partial_smiles_sequences

files = [
    "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-train-with-valid-all.jsonl",
    "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-test-all.jsonl",
]

for p in partial_smiles_sequences("O=C1CCC(=O)N1Br"):
    print(p)

tokenizer = BasicSmilesTokenizer()

BASELINE_TARGETS_COUNT = defaultdict(int)
BASELINE_TARGETS_SCORE = defaultdict(int)

# files = ["/Users/ato/Desktop/Git/rxn_negative_learning/dummy.jsonl"]
for ff in files:
    with open(ff, "r") as f:
        data = [json.loads(line.strip()) for line in f]
        for elem in data:
            for p in partial_smiles_sequences(" ".join(tokenizer.tokenize(elem["target"]))):
                BASELINE_TARGETS_COUNT[p] += 1
                BASELINE_TARGETS_SCORE[p] += elem["score"]

BASELINE_TARGETS_DICT = {}
for elem in BASELINE_TARGETS_COUNT.keys():
    BASELINE_TARGETS_DICT[elem] = BASELINE_TARGETS_SCORE[elem] / float(BASELINE_TARGETS_COUNT[elem])

print(BASELINE_TARGETS_DICT)

with open("/data/old/additional/baseline_targets.json", "w") as f:
    json.dump(BASELINE_TARGETS_DICT, f)

# print(len(POS_TARGETS_DICT.keys()))
#
# import json
#
# files = [
#     "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-train-with-valid-all.jsonl",
#     "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-test-all.jsonl"
# ]
# NEG_TARGETS_DICT = {}
# for ff in files:
#     with open(ff, "r") as f:
#         data = [json.loads(line.strip()) for line in f]
#         for elem in data:
#             if elem["score"] == 1:
#                 NEG_TARGETS_DICT[elem["idx"]] = elem["opposite_targets"]
#
# print(NEG_TARGETS_DICT)
# print(len(NEG_TARGETS_DICT.keys()))
