# coding: UTF-8
import torch
from sklearn import metrics
import time
import json
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import get_time_dif, gettoken
from loguru import logger


def train(
        writer,
        config,
        model,
        loss_fct,
        train_iter, dev_iter,
    ):
    total_steps = len(train_iter) * config.num_epochs
    start_time = time.time()
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3, correct_bias=False)

    if config.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=total_steps,
            num_warmup_steps=config.num_warmup_steps,
        )
    else:
        scheduler = None

    total_batch = 0  # 记录进行到多少batch
    dev_best_score = -1e12
    model.train()
    for epoch in range(config.num_epochs):
        logger.info(f"LR:{float(scheduler.get_last_lr()[0])}")
        writer.add_scalar("LR", float(scheduler.get_last_lr()[0]), epoch)
        logger.info('[Train] Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batches in enumerate(train_iter):
            model.zero_grad()
            sent, _, labels = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), \
                labels.to(config.device)
            position_ids = position_ids.to(config.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = loss_fct(pmi, labels)
            if (i + 1) % 10 == 1:
                logger.info(f"[Train] Epoch [{epoch + 1}/{config.num_epochs}], Batch [{i + 1}/{len(train_iter)}], "
                            f"Loss: {loss.item()}, LR: {float(scheduler.get_last_lr()[0])}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_batch += 1
            if total_batch % config.test_batch == 1:
                time_dif = get_time_dif(start_time)
                f1, _, dev_loss, predict_0, predict_1, ground, sents = evaluate(writer, loss_fct, epoch, i, len(train_iter), config, model,
                                                                   dev_iter, test=False)
                writer.add_scalar('TrainLoss', loss.item(), epoch * len(train_iter) + i)
                writer.add_scalar("Val Loss", dev_loss, epoch * len(train_iter) + i)
                logger.info(f"[Test] Dev loss: {dev_loss} Dev f1: {f1}, Time_dif: {time_dif}")
                if f1 > dev_best_score:
                    logger.info("[Save Model] Model saved, Dev_loss:%f, F1 score: %f" % (dev_loss, f1))
                    torch.save(model.state_dict(),
                               config.save_path + f"model_{str(config.bert_path).split('/')[-1]}_"
                                                  f"max_length_{config.max_length}_"
                                                  f"lr_{config.learning_rate}_"
                                                  f"threshold_{config.threshold}_"
                                                  # f"loss_weight_{config.loss_weight}_"
                                                  f"focal_loss_alpha_0.25_gama_2_"
                                                  f"dropout_{config.dropout}_"
                                                  f"neg_sample_{config.neg_sample}.ckpt")
                    dev_best_score = f1
                model.train()




def evaluate(writer, loss_fct, epoch, i, n, config, model, data_iter, test=True):
    # model.eval()
    loss_total = 0
    predicts_0, predicts_1, sents, grounds, all_bires, id_ = [], [], [], [], [], []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            sent, ids, labels = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(
                    config.device)
            position_ids = position_ids.to(config.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = loss_fct(pmi, labels)
            loss_total += loss.item()
            bires = torch.max(pmi.data, 1)[1]
            for b, g, p_0, p_1, s, ii in zip(bires, labels, pmi.data[:,0],pmi.data[:,1],sent, ids):
                all_bires.append(b.item())
                predicts_0.append(p_0.item())
                predicts_1.append(p_1.item())
                grounds.append(g.item())
                sents.append(s)
                id_.append(ii)
    accuracy = metrics.accuracy_score(grounds, all_bires)
    p = metrics.precision_score(grounds, all_bires, zero_division=0)
    r = metrics.recall_score(grounds, all_bires, zero_division=0)
    f1 = metrics.f1_score(grounds, all_bires, zero_division=0)
    writer.add_scalar("F1 score", f1, epoch * n + i)
    writer.add_scalar("Precision", p, epoch * n + i)
    writer.add_scalar("Recall", r, epoch * n + i)
    logger.info(
        f"[Test] F1 score: {f1}, Recall: {r}, Precision: {p}, Accuracy: {accuracy}")
    writer.add_pr_curve('positive pr curve', np.array(grounds), np.array(predicts_0), global_step=epoch * n + i)
    writer.add_pr_curve('negative pr curve', np.array(grounds), np.array(predicts_1), global_step=epoch * n + i)
    return f1, pmi, loss_total / len(data_iter), predicts_0, predicts_1, grounds, sents


def predict(config, model, data_iter):
    predicts = []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            sent, triple_id, _ = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
            input_ids, attention_mask, type_ids = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device)
            position_ids = position_ids.to(config.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            bires = torch.max(pmi.data, 1)[1]
            for b, t in zip(bires, triple_id):
                predicts.append({"salience": b.item(), "triple_id": t})
    with open(config.save_path + "xx_result.jsonl", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path + "model.ckpt"))
    model.eval()
    start_time = time.time()
    predict(config, model, test_iter)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:", time_dif)
