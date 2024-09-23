# LogAnalyzer V1 : Log analysis and classification using Transformers

## Introduction

This is an implementation of a system log classifier to identify the success or error scenarios. The dataset is locally curated and preprocessed by filtering out key log keywords and phrases. The implentation uses a fine-tuned version of **Distilroberta** to perform a sequence classification task. A custom Dataset class is used for loading the local dataset in csv format.

Training Metrics  
```
  {'eval_loss': 0.0007737778360024095, 'eval_accuracy': 1.0, 'eval_runtime': 0.021, 'eval_samples_per_second': 238.099, 'eval_steps_per_second': 47.62, 'epoch': 30.0}   
  {'train_runtime': 47.6537, 'train_samples_per_second': 69.25, 'train_steps_per_second': 8.814, 'train_loss': 0.6937108271799627, 'epoch': 30.0}   
  ***** train metrics *****   
    epoch                    =       30.0   
    total_flos               =   129617GF   
    train_loss               =     0.6937   
    train_runtime            = 0:00:47.65   
    train_samples_per_second =      69.25   
    train_steps_per_second   =      8.814   
  100%|██████████| 1/1 [00:00<00:00, 55.59it/s]   
  ***** eval metrics *****   
    epoch                   =       30.0   
    eval_accuracy           =        1.0   
    eval_loss               =     0.0008   
    eval_runtime            = 0:00:00.02   
    eval_samples_per_second =    250.009   
    eval_steps_per_second   =     50.002   
  Predicted class: ERROR_RES   
```

## Author : Kuljeet Singh