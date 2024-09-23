# Description : Linux System Log  Classification using Roberta
#   This is version 1 : Simpler approach , Version 2 will provide more finer control
# Date : 9/6/2024 (06)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#   P1: Create custom dataset
#   S1: Load files from disk and have encodings and labels features
# Revisions :
#

import numpy as np
import pandas as pd
import os
import sys

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from SyslogsDataset import LogDataset


DATA_PATH = "data"
TRAIN_FILE = os.path.join(DATA_PATH, "LOG_TRAIN.csv")
TEST_FILE = os.path.join(DATA_PATH, "LOG_TEST.csv")
SAMPLE_TEST = os.path.join(DATA_PATH, "Sample_TEST.csv")
MODEL_NAME = "distilroberta-base"
SAVED_MODEL = "distilroberta-logclassifier"
# MODEL_NAME = 'roberta-base'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_n_eval():
    metric = load_metric("accuracy")

    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = LogDataset(TRAIN_FILE, tokenizer=tokenizer)
    print(train_dataset[0])
    #   train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = LogDataset(
        TEST_FILE, tokenizer=tokenizer, classes=train_dataset.classes
    )
    print(test_dataset)
    #   test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print(train_dataset.num_classes)
    labels = train_dataset.get_classes()
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=train_dataset.id2label,  # Add to Model Config
        label2id=train_dataset.label2id,  # Add to Model Config
    )

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        eval_strategy="steps",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,  # use to save best model later
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  # Capture accuracy metric
    )

    train_results = trainer.train()
    trainer.save_model(SAVED_MODEL)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()  # This is important - model cannot load without this
    # Evaluate model
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)


def predict_one_log(log_filename):
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)  # use the base model
    model = RobertaForSequenceClassification.from_pretrained(
        SAVED_MODEL, local_files_only=True
    )
    testds = LogDataset(SAMPLE_TEST, tokenizer=tokenizer)
    markers = testds.markers
    encodings = tokenizer(markers, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**encodings)
    logits = outputs.logits  # For classification  task
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    print("Python %s on %s" % (sys.version, sys.platform))
    train_n_eval()
    predict_one_log(SAMPLE_TEST)
