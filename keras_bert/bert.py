# coding=utf-8
import numpy as np

from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def get_representations(self, input_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True
        )
        pooled_output = self.dropout(pooled_output)
        return encoded_layers, pooled_output

    def get_linear_weights(self):
        return self.classifier.state_dict()


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            return loss
        else:
            return logits

    def get_representations(self, input_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True
        )
        pooled_output = self.dropout(pooled_output)
        return encoded_layers, pooled_output

    def get_linear_weights(self):
        return self.classifier.state_dict()


class BertForMultiTaskSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, tasks_num_labels, device):
        super(BertForMultiTaskSequenceClassification, self).__init__(config)
        self.tasks_num_labels = tasks_num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = []
        for num_labels in tasks_num_labels:
            self.classifiers.append(
                nn.Linear(config.hidden_size, num_labels).to(device)
            )
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        tasks=None,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        tasks_logits = []
        for i, classifier in enumerate(self.classifiers):
            tasks_logits.append(classifier(pooled_output))
        tasks_logits = np.array(tasks_logits)
        if labels is not None and tasks is not None:
            loss_fct = CrossEntropyLoss()
            mean_loss = None
            weights = [0.5] * len(self.classifiers)
            for task_num in range(len(self.classifiers)):
                task_indexes = np.array(tasks.cpu().numpy() == task_num)
                if len(np.where(task_indexes)[0]) == 0:
                    continue
                task_labels = labels[np.where(task_indexes)[0]]
                logits = tasks_logits[task_num][np.where(task_indexes)[0]]
                loss = loss_fct(
                    logits.view(-1, self.tasks_num_labels[task_num]),
                    task_labels.view(-1),
                )
                if mean_loss is None:
                    mean_loss = weights[task_num] * loss
                else:
                    mean_loss += weights[task_num] * loss
            return mean_loss
        else:
            return tasks_logits

    def get_linear_weights(self):
        return [classifier.state_dict() for classifier in self.classifiers]
