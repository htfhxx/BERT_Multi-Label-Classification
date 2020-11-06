import torch
import torch.nn as nn
from transformers import BertModel




class BertClassifier(nn.Module):
    def __init__(self, device, transformer_width):
        super(BertClassifier, self).__init__()
        self.device = device
        self.bert_layer = BertModel.from_pretrained("checkpoints/bert-base-chinese")  #
        self.classifier_layer = nn.Linear(transformer_width, 12)


    def forward(self, x):
        _, y = self.bert_layer(x)
        logits = self.classifier_layer(y)

        return logits