import csv
import random

from scorer import load_dataset, score_submission, print_confusion_matrix


def get_random_stance():
    # Higher probability to use unrelated, as we expect more of them
    if random.randint(0, 10) < 9:
        return 'unrelated'
    else:
        return 'discuss'


with open("datasets/stances/basic_competition_stances_unlabeled.csv") as dataset:
    reader = csv.DictReader(dataset)
    data = list(reader)

for entry in data:
    entry['Stance'] = get_random_stance()

gold_data = load_dataset("datasets/stances/basic_competition_stances.csv")

test_score, cm = score_submission(gold_data, data)
print_confusion_matrix(cm)
