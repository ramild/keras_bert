import csv
import os
import sys
import pandas as pd


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, task=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task = task


def create_examples(texts, labels, split_name=None, task=None):
    examples = []
    assert len(texts) == len(labels), "Shapes do not match"
    for i, text in enumerate(texts):
        if split_name:
            guid = "{}-{}".format(split_name, i)
        else:
            guid = "{}".format(i)
        if text != text:
            text = ""
        examples.append(
            InputExample(
                guid=guid, text_a=text, text_b=None, label=labels[i], task=task
            )
        )
    return examples


def create_multitask_examples(texts, labels=None, split_name=None):
    examples = []
    assert len(texts) == len(labels), "Shapes do not match"
    for task_num, text in enumerate(texts):
        examples += create_examples(
            texts[task_num], labels[task_num], split_name, task=task_num
        )
    return examples
