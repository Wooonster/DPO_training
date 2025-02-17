import torch
import torch.nn.functional as F
from dataset import DPODataset, DPODataCollator
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from train import LLM, Config

def logits_to_probs(logits, labels):
    '''
    将 logits 转换为 概率
    args:
        logits: -> (batch_size, seq_len, vocab_size)
        labels: -> (batch_size, seq_len)

    returns:
        prob: -> (batch_size, seq_len)
    '''
    # 对 logits 进行 log softmax 操作，得到 log 概率
    log_probs = F.softmax(logits, dim=2)
    # 使用 labels 作为索引，从 log_probs 中提取对应的 log 概率，并去掉最后一维
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)

    masked_probs = []
    for prob, label in zip(probs, labels):
        masked_probs.append(probs[label != 0].sum().unsqueeze(0))
    return masked_probs

def dpo_loss(probs, ref_probs, beta=0.1):
    def split_probs(prob):
        len_chosen = int(len(probs)) // 2
        chosen, reject = prob[:len_chosen], prob[len_chosen:]
        return torch.cat(chosen), torch.cat(reject)

    chosen, reject = split_probs(probs)
    ref_chosen, ref_reject = split(ref_probs)

    loss = -F.logsigmoid(beta * ((chosen - reject) - (ref_chosen - ref_reject)))
    return loss.mean()

class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels = inputs['input_ids'], inputs['labels']
        
        # reference prob
        # 冻结 reference model
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)

        # model prob
        logits = model(input_ids=input_ids, labels=labels).logits
        probs = logits_to_probs(logits, labels)

        loss = dpo_loss(probs, ref_probs, 0.1)
        return loss

