# coding: UTF-8
import json
import time
import random
import argparse
import numpy as np
from loguru import logger
from config import Config
from train_eval import train

import torch
import torch.nn as nn

from models import KGBert, PromptBert
from utils import get_time_dif, load_dataset, gettoken
from torch.utils.data import DataLoader

from loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Salient triple classification')
parser.add_argument("--do_train", action="store_true", help="Whether to run training.",)
parser.add_argument("--do_test", action="store_true", help="Whether to run testing.",)
parser.add_argument("--test_batch", default=200, type=int, help="Test every X updates steps.")
parser.add_argument("--device", default="cuda:0", type=str, help="device")
parser.add_argument("--data_dir", default="data", type=str, help="The task data directory.")
parser.add_argument("--model_dir", default="bert-base-chinese", type=str, help="The directory of pretrained models")
parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
# models param
parser.add_argument("--max_length", default=128, type=int, help="the max length of sentence.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")
parser.add_argument('--threshold', type=float, default=0.4)
parser.add_argument('--loss_weight', type=list, default=[2.0, 1.0])
parser.add_argument('--train_dataset', type=str, default='/train_triple_aug.jsonl')
args = parser.parse_args()




def train_entry(writer, config, model, criterion):
    start_time = time.time()
    logger.info("Loading data...")
    train_data_all = load_dataset(config.train_path, config)
    random.shuffle(train_data_all)
    offset = int(len(train_data_all) * 0.1)
    dev_data = train_data_all[:offset]
    train_data = train_data_all[offset:]
    train_iter = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=False)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:", time_dif)
    train(writer, config, model, criterion, train_iter, dev_iter)


def test_entry(model, config):
    test_data = load_dataset(config.test_path, config)
    model.load_state_dict(torch.load(config.save_path+"model_RoBERTa_zh_Large_max_length_256_lr_1e-05threshold_0.7_loss_weight_[1.0, 0.7]_dropout_0.1_neg_sample_2.ckpt"))
    model.eval()
    loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    predicts = []
    for i, batches in enumerate(loader):
        sent, triple_id, _ = batches
        input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
        input_ids, attention_mask, type_ids = \
            input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device)
        position_ids = position_ids.to(config.device)
        pmi = model(input_ids, attention_mask, type_ids, position_ids)
        bires = torch.max(pmi.data, 1)[1]
        for b, t in zip(bires, triple_id):
            predicts.append({"salience": b.item(), "triple_id": t})

    with open(config.save_path + "gwal_result.jsonl", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")
        logger.info("保存预测文件")



def main():
    config = Config(args)
    logger.info("训练参数：")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.are_deterministic_algorithms_enabled = True

    if config.loss_type == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss_type == "FL":
        criterion = FocalLoss()
    else:
        raise NotImplementedError

    # Define model
    if config.model_name == "KGBert":
        model = KGBert(config).to(config.device)
    elif config.model_name == "PromptBert":
        model = PromptBert(config).to(config.device)
    else:
        raise NotImplementedError

    if config.do_train:
        writer = SummaryWriter(f"./runs/{str(config.bert_path).split('/')[-1]}_"
                               f'max_length_{config.max_length}_'
                               f'lr_{config.learning_rate}_'
                               # f'loss_weight_{config.loss_weight}_'
                               f"focal_loss_alpha_0.25_gama_2_"
                               f'dropout_{config.dropout}_'
                               f'threshold_{config.threshold}_'
                               f'neg_sample_{config.neg_sample}')
        train_entry(writer, config=config, model=model, criterion=criterion)
        print("开始训练")

    if config.do_test:
        print("开始测试")
        test_entry(model=model, config=config)




if __name__ == '__main__':
    main()