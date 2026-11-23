
import json
import random
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "employee-handbook-rag"
    / "data"
    / "evaluation"
    / "evaluation.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "balanced_evaluation_sample.jsonl"
)

RANDOM_SEED = 42

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            data.append(json.loads(line))

# --------------------------------------------------
# 2. Group by question type
# --------------------------------------------------

groups = {
    "factual": [],
    "policy": [],
    "paraphrased": [],
    "unanswerable": []
}

for item in data:

    question_type = item.get("question_type")

    if question_type not in groups:
        raise ValueError(
            f"Unknown question type: {question_type}"
        )

    groups[question_type].append(item)

# --------------------------------------------------
# 3. Sample questions
# --------------------------------------------------

sample_sizes = {
    "factual": 60,
    "policy": 10,
    "paraphrased": 5,
    "unanswerable": 10
}

rng = random.Random(RANDOM_SEED)

sample = []

for group, required_count in sample_sizes.items():

    available = groups[group]

    selected_count = min(
        required_count,
        len(available)
    )

    selected = rng.sample(
        available,
        selected_count
    )

    sample.extend(selected)

# --------------------------------------------------
# 4. Shuffle sample
# --------------------------------------------------

rng.shuffle(sample)

# --------------------------------------------------
# 5. Save sample
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    for item in sample:

        file.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

# --------------------------------------------------
# 6. Validate and display
# --------------------------------------------------

type_counts = Counter(
    item["question_type"]
    for item in sample
)

difficulty_counts = Counter(
    item["difficulty"]
    for item in sample
)

answerability_counts = Counter(
    item["answerable"]
    for item in sample
)

print("\n========== BENCHMARK CREATED ==========")

print("Total questions:", len(sample))

print("\nQuestion type distribution:")
for key, value in sorted(type_counts.items()):
    print(f"{key}: {value}")

print("\nDifficulty distribution:")
for key, value in sorted(difficulty_counts.items()):
    print(f"{key}: {value}")

print("\nAnswerability distribution:")
for key, value in sorted(answerability_counts.items()):
    print(f"{key}: {value}")

print("\nSaved to:")
print(OUTPUT_FILE)
