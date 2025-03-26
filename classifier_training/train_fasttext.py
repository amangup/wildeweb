from datasets import load_dataset

import string
import fasttext

DATASET_NAME = "amang1802/wildeweb_cls_labels_v1"

def text_simplify(text):
    no_punct = ''.join(ch for ch in text[:5000] if ch not in string.punctuation)
    return no_punct.replace("\n", " ").lower()


def dataset_to_train_test_txt(dataset, binary=False):
    label_col = 'label' if binary else 'binary_label'
    ds_txt = dataset.map(lambda row: {"fasttext": f"__label__{row[label_col]} {text_simplify(row['text'])}"})

    train_txt = "\n".join(ds_txt['train']['fasttext'])
    test_txt = "\n".join(ds_txt['test']['fasttext'])

    dataset_title = DATASET_NAME.split("/")[1]

    train_filename = f"{dataset_title}_train.txt"
    test_filename = f"{dataset_title}_test.txt"

    with open(train_filename, "w") as f:
        f.write(train_txt)

    with open(test_filename, "w") as f:
        f.write(test_txt)

    return train_filename, test_filename


def print_results(N, p, r, f1=-1.0):
    print(f"N\t{N}")
    print(f"P@1\t{p:.3f}")
    print(f"R@1\t{r:.3f}")
    print(f"F1@1\t{f1:.3f}")

def print_label_results(results):
    for k, v in sorted(results.items(), key=lambda item: item[0]):
        print("\n" + k)
        print_results(0, v['precision'], v['recall'], v['f1score'])

def main():
    dataset = load_dataset(DATASET_NAME)
    train_filename, test_filename = dataset_to_train_test_txt(dataset)
    model = fasttext.train_supervised(train_filename, epoch=10, lr=0.2, wordNgrams=2, dim=200)

    print_results(*model.test(test_filename))
    print_label_results(model.test_label(test_filename))


if __name__ == "__main__":
    main()