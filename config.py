import torch
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM

class Config(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'KGBert' # PromptBert
        self.rank = -1
        self.local_rank = -1
        self.train_path = args.data_dir + args.train_dataset  # 训练集
        self.test_path = args.data_dir + '/dev_triple.jsonl'  # 测试集
        self.save_path = args.output_dir  # 模型训练结果
        self.bert_path = args.model_dir
        self.test_batch = args.test_batch
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_workers = 1
        self.local_rank = -1
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                           # mini-batch大小
        self.learning_rate = args.learning_rate                                     # 学习率
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, do_lower_case=True)
        self.hidden_size = args.hidden_size
        self.max_length = args.max_length
        self.do_train = args.do_train
        self.do_test = args.do_test
        self.do_aug = False
        self.neg_sample = 2
        self.loss_weight = args.loss_weight
        self.threshold = args.threshold
        self.num_warmup_steps = 2000
        self.loss_type = "FL" # BCE/FL
        self.use_scheduler = True