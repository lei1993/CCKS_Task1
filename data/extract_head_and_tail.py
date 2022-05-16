import json
from tqdm import tqdm
import csv
import random

def extract(train_file_path):
    sub_set = set()
    obj_set = set()
    pred_set = set()

    with open(train_file_path, 'r',encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            subject = line_dict['subject']
            object = line_dict['object']
            predicate = line_dict['predicate']
            if subject in sub_set:
                continue
            else:
                sub_set.add(subject)
            if object in obj_set:
                continue
            else:
                obj_set.add(object)
            if predicate in pred_set:
                continue
            else:
                pred_set.add(predicate)
    write_data('subjects.txt', sub_set)
    write_data('objects.txt', obj_set)
    write_data('relations.txt', pred_set)



def write_data(name, set_data):
    f = open(name, 'w')
    for item in set_data:
        f.write(item+'\n')
    f.close()


SEP = '[SEP]'
def write_tsv(path, out_name):
    rows = []
    pos_samples = []
    neg_samples = []
    spli_ratio= 0.1
    train_file = open('train_ccks.tsv', 'w', newline='')
    valid_file = open('valid_ccks.tsv', 'w', newline='')
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            sid = line_dict['triple_id']
            subject = line_dict['subject']
            object = line_dict['object']
            predicate = line_dict['predicate']
            salicence = line_dict['salience']
            if salicence == '1':
                pos_samples.append([sid, SEP.join([subject, predicate, object]), salicence])
            if  salicence == '0':
                neg_samples.append([sid, SEP.join([subject, predicate, object]), salicence])
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        valid_pos_samples, valid_neg_samples = \
            pos_samples[0:int(spli_ratio * len(pos_samples))], neg_samples[0:int(spli_ratio * len(neg_samples))]
        train_pos_samples, train_neg_samples = \
            pos_samples[int(spli_ratio * len(pos_samples)):], neg_samples[:int(spli_ratio * len(neg_samples))]
        print(f"Train samples in total: \n positive: {len(train_pos_samples)}, negative: {len(train_neg_samples)}")
        print(f"Valid samples in total: \n positive: {len(valid_pos_samples)}, negative: {len(valid_neg_samples)}")
        train_samples = train_pos_samples + train_neg_samples
        random.shuffle(train_samples)
        valid_samples = valid_pos_samples + valid_neg_samples
        random.shuffle(valid_samples)
        tw = csv.writer(train_file, delimiter='\t')
        tw.writerows(train_samples)
        tw1 = csv.writer(valid_file, delimiter='\t')
        tw1.writerows(valid_samples)

def write_test_tsv():
    rows = []
    test_file = open('test_ccks.tsv', 'w', newline='')
    with open('./dev_triple.jsonl', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            sid = line_dict['triple_id']
            subject = line_dict['subject']
            object = line_dict['object']
            predicate = line_dict['predicate']
            rows.append([sid, SEP.join([subject, predicate, object])])

        tw1 = csv.writer(test_file, delimiter='\t')
        tw1.writerows(rows)



if __name__ == '__main__':
    # path = './train_triple.jsonl'
    # extract(path)
    write_tsv('./train_triple.jsonl', 'train')
    write_tsv('./dev_triple.jsonl', 'dev')
    # write_test_tsv()
