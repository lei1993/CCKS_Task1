# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel

class KGBert(nn.Module):
    def __init__(self, config, ):
        super(KGBert, self).__init__()
        self.mid_size = 512
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
                nn.Linear(config.hidden_size, self.mid_size),
                nn.BatchNorm1d(self.mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.mid_size, self.num_labels)
            )

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=type_ids,
            position_ids=position_ids,)
        cls_embed = outputs[1]
        output = self.dropout(cls_embed)
        x = self.fc(output)
        return x




class PromptBert(nn.Module):
    def __init__(self, config, ):
        super(PromptBert, self).__init__()
        self.mid_size = 512
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(config.hidden_size, 1)
        self.fc = nn.Sequential(
                nn.Linear(config.hidden_size, self.mid_size),
                nn.BatchNorm1d(self.mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.mid_size, self.num_labels)
            )

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=type_ids,
            position_ids=position_ids,)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        # x = self.fc(torch.mean(sequence_output, 1))
        # x = torch.sigmoid(x).squeeze(-1)
        x = self.fc(sequence_output)
        return x




