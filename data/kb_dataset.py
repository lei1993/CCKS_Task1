import torch
from torch.utils.data import Dataset
import collections
import json
import numpy as np
import random
from os.path import join
import os
from uuid import uuid4


def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


class KbDataset(Dataset):

    @staticmethod
    def build_graph(raw_examples):
        # build positive graph from triples
        subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
        obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))

        for _raw_ex in raw_examples:
            _head, _rel, _tail = _raw_ex[:3]
            subj2objs[_head][_rel].add(_tail)
            obj2subjs[_tail][_rel].add(_head)
        return subj2objs, obj2subjs

    @staticmethod
    def build_type_constrain_dict(raw_examples):
        type_constrain_dict = collections.defaultdict(lambda: {"head": [], "tail": []})
        for _raw_ext in raw_examples:
            _head, _rel, _tail = _raw_ext[:3]
            type_constrain_dict[_rel]["head"].append(_head)
            type_constrain_dict[_rel]["tail"].append(_tail)
        return type_constrain_dict

    def update_negative_sampling_graph(self, raw_examples):
        # graph
        for _raw_ext in raw_examples:
            _head, _rel, _tail = _raw_ext[:3]
            self.subj2objs[_head][_rel].add(_tail)
            self.obj2subjs[_tail][_rel].add(_head)

        if self.pos_triplet_str_set is not None:
            self.pos_triplet_str_set.update(set(self._triple2str(_ext) for _ext in raw_examples))

    def pre_negative_sampling(self, pos_raw_examples, neg_times):
        neg_raw_example_lists = []
        for _pos_raw_ex in pos_raw_examples:
            neg_raw_example_list = []
            for _ in range(neg_times):
                neg_raw_example_list.append(self.negative_sampling(_pos_raw_ex, self.neg_weights))
            neg_raw_example_lists.append(neg_raw_example_list)
        return neg_raw_example_lists

    def __init__(self, data_path="/data/zlei/nlp/OpenBG/OpenBG-CSK/data", data_type="train", do_lower_case=True,
                 tokenizer_type="bert", tokenizer=None,
                 neg_times=0, neg_weights=None, *args, **kwargs):
        self.data_path = data_path
        self.raw_examples = self._read_raw_examples(self.data_path)
        self.data_type = data_type
        self.subj2objs, self.obj2subjs = None, None
        self.neg_times = neg_times
        self.neg_weights = neg_weights
        self.do_lower_case = do_lower_case
        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer

        assert self.tokenizer_type in ["bert", "roberta"]

        if self.data_type == "train":
            self.subj2objs, self.obj2subjs = self.build_graph(self.raw_examples)

        self.pos_triplet_str_set = set(self._triple2str(_ex) for _ex in self.raw_examples)

        self.ent_list, self.rel_list, self.ent2text, self.rel2text = self._read_ent_rel_info()
        self.ent2idx = dict((_e, _idx) for _idx, _e in enumerate(self.ent_list))
        self.rel2idx = dict((_e, _idx) for _idx, _e in enumerate(self.rel_list))

        self._sep_id, self._cls_id, self._pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token, self.tokenizer.cls_token, self.tokenizer.pad_token, ]
        )

        # if needed, pre-sampled negative examples
        self.negative_samples = None
        # for some special dataset which needs pre-defined text
        self.triplet2text = None

    def _read_ent_rel_info(self):
        ent_list = self.load_list_from_file(join(self.data_path, "entities.txt"))
        rel_list = self.load_list_from_file(join(self.data_path, "relations.txt"))
        # read entities and relations's text from file
        ent2text = dict(
            tuple([_line.strip()] * 2) for _line in self.load_list_from_file(join(self.data_path, "entities.txt")))
        rel2text = dict(
            tuple([_line.strip()] * 2) for _line in self.load_list_from_file(join(self.data_path, "relations.txt")))
        return ent_list, rel_list, ent2text, rel2text

    @staticmethod
    def _triple2str(raw_kg_triplet):
        return '\t'.join(raw_kg_triplet[:3])

    def __getitem__(self, idx):
        if self.data_type == "train":  # this is for negative sampling
            assert self.subj2objs is not None and self.obj2subjs is not None

        if self.neg_times > 0:
            pos_item = idx // (1 + self.neg_times)
            if idx % (1 + self.neg_times) == 0:
                label = 1
                raw_ex = self.raw_examples[pos_item]
            else:
                label = 0
                if self.negative_samples is None:
                    pos_raw_ex = self.raw_examples[pos_item]
                    raw_ex = self.negative_sampling(pos_raw_ex, self.neg_weights)
                else:
                    neg_exs = self.negative_samples[pos_item]
                    if isinstance(neg_exs, list) and len(neg_exs) in [3, 4] \
                            and all(isinstance(_e, (str, int)) for _e in neg_exs):
                        raw_ex = neg_exs
                    else:
                        neg_idx = idx % (1 + self.neg_times) - 1  # from [1,self.neg_times] -> [0, self.neg_times-1]
                        raw_ex = neg_exs[neg_idx % len(neg_exs)]
        else:
            raw_ex = self.raw_examples[idx]

            if len(raw_ex) > 3:  # 为什么会大于3？  训练集的label哪来的？
                label = int(float(raw_ex[3]) > 0)
            elif self.data_type in ["dev", "test"]:  # if no label in "dev", "test", default is "positive"
                label = 1
            else:
                raise AttributeError
            raw_ex = raw_ex[:3]

        return raw_ex, label

    def __len__(self):
        return len(self.raw_examples) * (1 + self.neg_times)

    @staticmethod
    def _read_raw_examples(data_path):
        examples = []
        lines = load_json(join(data_path, 'train_triple.jsonl'))
        for _idx, line in enumerate(lines):
            line = json.loads(line)
            if line['salience'] == '1':
                examples.append([line['subject'], line['predicate'], line['object']])
        return examples

    def negative_sampling(self, raw_kg_triplet, weights=None, *args, **kwargs):
        head, rel, tail = raw_kg_triplet[:3]
        if weights is None:
            weights = [1., 1., 1.]
        cdf = np.cumsum(np.array(weights) / sum(weights))

        prob = random.random()

        neg_example = [head, rel, tail]

        if self.data_type == "train":
            while True:
                if prob < cdf[0]:
                    src_elem = neg_example[0]
                    while True:
                        rdm_elem = random.choice(self.ent_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[0] = rdm_elem
                elif prob < cdf[1]:
                    src_elem = neg_example[2]
                    while True:
                        rdm_elem = random.choice(self.ent_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[2] = rdm_elem
                else:
                    src_elem = neg_example[1]
                    while True:
                        rdm_elem = random.choice(self.rel_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[1] = rdm_elem
                if self.pos_triplet_str_set is not None and self._triple2str(neg_example) in self.pos_triplet_str_set:
                    continue
                else:
                    break
            return neg_example

        while True:
            if prob < cdf[0]:
                pos_ent_list = self.obj2subjs[tail][rel]
                pos_ent_list = set(pos_ent_list)
                neg_elem = None
                max_iter = 1000
                while max_iter > 0:
                    neg_elem = random.choice(self.ent_list)
                    if neg_elem not in pos_ent_list:
                        break
                    max_iter -= 1
                if max_iter == 0:
                    print("Warning: max iter reached when negative sampling, chose from pos set")
                assert neg_elem is not None
                neg_example[0] = neg_elem

            elif prob < cdf[1]:
                # do for tail entity
                pos_ent_list = self.subj2objs[head][rel]
                pos_ent_list = set(pos_ent_list)
                neg_elem = None
                max_iter = 1000
                while max_iter > 0:
                    neg_elem = random.choice(self.ent_list)
                    if neg_elem not in pos_ent_list:
                        break
                    max_iter -= 1
                if max_iter == 0:
                    print("Warning: max iter reached when negative sampling, chose from pos set")

                assert neg_elem is not None
                neg_example[2] = neg_elem
            else:
                src_elem = neg_example[1]
                while True:
                    rdm_elem = random.choice(self.rel_list)
                    if src_elem != rdm_elem:
                        break
                    assert rdm_elem is not None
                    neg_example[1] = rdm_elem
                if self.pos_triplet_str_set is not None and self._triple2str(neg_example) in self.pos_triplet_str_set:
                    neg_example = [head, rel, tail]
                    continue
                else:
                    break
            return neg_example

    def str2ids(self, text, max_len=None):
        if self.do_lower_case and self.tokenizer_type == "bert":
            text = text.lower()
        text = self.tokenizer.cls_token + " " + text
        wps = self.tokenizer.tokenize(text)
        if max_len is not None:
            wps = self.tokenizer.tokenize(text)[:max_len]
        wps.append(self.tokenizer.sep_token)
        return self.tokenizer(wps)

    def convert_raw_example_to_features(self, raw_kg_triplet, method="0"):
        head, rel, tail = raw_kg_triplet[:3]

        if method == "0":
            head_ids = self.str2ids("Node: " + self.ent2text[head])
            relidx = self.rel2idx[rel]
            tail_ids = self.str2ids(("Node: " + self.ent2text[tail]))
            return head_ids, relidx, tail_ids
        elif method == "1":
            head_ids = self.str2ids("Node: " + self.ent2text[head])
            rel_ids = self.str2ids("Rel: " + self.rel2text[rel])
            tail_ids = self.str2ids("Node: " + self.ent2text[tail])
            return head_ids, rel_ids, tail_ids
        elif method == "2":
            # combine
            rel_ids = self.str2ids(self.rel2text[rel])[1:-1]  # [1:-1] for removing special token from self.str2ids
            remain_len = self.max_seq_length - 4 - len(rel_ids)
            assert remain_len >= 4  # need sufficient budget for entities' ids
            head_ids = self.str2ids(self.ent2text[head])[1:-1]
            tail_ids = self.str2ids(self.ent2text[tail])[1:-1]
            while len(head_ids) + len(tail_ids) > remain_len:
                if len(head_ids) > len(tail_ids):
                    head_ids.pop(-1)
                else:
                    tail_ids.pop(-1)
            input_ids = [self._cls_id] + head_ids + [self._sep_id] + rel_ids + [self._sep_id] + tail_ids + [
                self._sep_id]
            type_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1) + [0] * (len(tail_ids) + 1)
            return input_ids, type_ids
        elif method == "3":
            assert self.triplet2text is not None
            triplet_str = "\t".join([head, rel, tail])
            gen_sent = self.triplet2text[triplet_str]
            input_ids = self.str2ids(gen_sent, self.max_seq_length)
            type_ids = [0] * len(input_ids)
            return input_ids, type_ids
        elif method == "4":
            head_ids = self.str2ids(self.ent2text[head])
            rel_ids = self.str2ids(self.rel2text[rel])
            tail_ids = self.str2ids(self.ent2text[tail])
            return head_ids, rel_ids, tail_ids
        elif method == "5":
            head_ids = self.str2ids(self.ent2text[head])[1:-1]
            rel_ids = self.str2ids(self.rel2text[rel])[1:-1]
            tail_ids = self.str2ids(self.ent2text[tail])[1:-1]
            return head_ids, rel_ids, tail_ids
        else:
            raise KeyError(method)

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t % 3 == 0:
                padding_value = self._pad_id
            else:
                padding_value = 0
            if _tensors[0].dim() >= 1:
                return_list.append(
                    torch.nn.utils.rnn.padd_sequence(_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)

    @staticmethod
    def load_list_from_file(file_path):
        data = []
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding='utf-8') as fp:
                for line in fp:
                    data.append(line.strip())
        return data


if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    tokenizer = BertTokenizer.from_pretrained('/data/zlei/nlp/OpenBG/OpenBG-CSK/prev_model/roberta')
    dataset = KbDataset('./', 'train', tokenizer=tokenizer, neg_times=2)
    loader = DataLoader(dataset, batch_size=1)
    print(len(loader))
    f = open('train_triple_aug.jsonl', 'w', encoding='utf-8', newline='\n')
    for item in loader:
        sub, pre, obj = item[0][:3]

        data = json.dumps({
            "triple_id": str(uuid4()).replace('-', ''),
            "subject": sub[0],
            "object": obj[0],
            "predicate": pre[0],
            "salience": str(item[1].item())}, ensure_ascii=False)
        f.write(data + '\n')
    # lines = load_json(join('./', 'train_triple.jsonl'))
    # for _idx, line in enumerate(lines):
    #     line = json.loads(line)
    #     if line['salience'] == '0':
    #         f.write(json.dumps(line, ensure_ascii=False) + '\n')
    f.close()
