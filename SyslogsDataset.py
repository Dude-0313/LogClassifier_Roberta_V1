# Description : Create a custom dataset for Syslog Classification
# Date : 9/6/2024 (06)
# Author : Dude
# URLs :
#           https://huggingface.co/transformers/v3.2.0/custom_datasets.html
# Problems / Solutions :
#
# Revisions :
#
import torch
from torch.utils.data import Dataset, DataLoader


class LogDataset(Dataset):
    def __init__(self, filename, tokenizer=None, classes=None):
        self.tokenizer = tokenizer
        if classes is None:
            self.markers, self.labels, self.classes = self.load_data(filename)
        else:
            self.markers, self.labels, _ = self.load_data(filename)
            self.classes = classes
        self.id2label = {str(i): c for i, c in enumerate(self.classes)}
        self.label2id = {c: str(i) for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.encodings = self.tokenizer(self.markers, padding=True, truncation=True)

    @staticmethod
    def load_data(csv_filename):
        log_markers = []
        log_class = []
        log_classes = set()
        with open(csv_filename, "r") as f:
            for line in f:
                label = line.rsplit(",")[-1].rstrip()
                text = " ".join(line.split(",")[:-1])
                log_markers.append(text)
                log_class.append(label)
                log_classes.add(label)
        return log_markers, log_class, log_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #        x = self.labels[idx]
        #        x = int(self.label2id[x])
        item["labels"] = torch.tensor(int(self.label2id[self.labels[idx]]))
        return item

    def get_classes(self):
        return self.classes
