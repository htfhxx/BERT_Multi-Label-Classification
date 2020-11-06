
from sklearn.metrics import f1_score,recall_score,precision_score
import numpy as np
import torch
import torch.nn as nn


def sequences_to_nhot(seqs, attr_vocab_size):
    labels = np.zeros((len(seqs), attr_vocab_size), dtype=np.int)

    for bid, seq in enumerate(seqs):
        for word in seq:
            if word>0:
                labels[bid][word-1] = 1
    return labels


class MultilabelScorer:
    def __init__(self):
        self.clear()


    def clear(self):
        self.prediciton_list = []
        self.reference_list = []


    def update(self, labels, prediction):
        self.prediciton_list.extend(labels)
        self.reference_list.extend(prediction)


    def get_avg_scores(self):
        micro_precision = precision_score(self.reference_list, self.prediciton_list, average="micro")
        micro_recall = recall_score(self.reference_list, self.prediciton_list,  average="micro")
        micro_f1 = f1_score(self.reference_list, self.prediciton_list, average="micro")

        macro_precision = precision_score(self.reference_list, self.prediciton_list,  average="macro")
        macro_recall = recall_score(self.reference_list, self.prediciton_list,  average="macro")
        macro_f1 = f1_score(self.reference_list, self.prediciton_list,  average="macro")

        weighted_precision = precision_score(self.reference_list, self.prediciton_list, average="weighted")
        weighted_recall = recall_score(self.reference_list, self.prediciton_list, average="weighted")
        weighted_f1 = f1_score(self.reference_list, self.prediciton_list,  average="weighted")

        return micro_precision,  micro_recall, micro_f1

    def print_avg_scores(self):
        avg_precisions,  avg_recalls, avg_F1 = self.get_avg_scores()
        print(f"Average micro precisions: {avg_precisions}")
        print(f"Average micro recalls: {avg_recalls}")
        print(f"Average micro f1: {avg_F1}")


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        logits = logits.contiguous()
        targets = targets.contiguous()
        sup_loss = self.BCE(logits.squeeze(1), targets).mean()
        return sup_loss
