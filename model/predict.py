import numpy as np
import torch

#from App import lord_label_dict
import torch.nn as nn

def lord_label_dict(path):
    label2id = {}
    id2label = {}
    f = open(path, "r", encoding="utf-8")
    for item in f:
        id, label = item.strip().split("\t")
        label2id[label] = int(id)
        id2label[int(id)] = label
    f.close()
    return id2label, label2id

def process_emb(embedding, emb_dim):
    embeddings = {}
    embeddings["<pad>"] = np.zeros(emb_dim)
    embeddings["<unk>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)
    embeddings["</s>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)
    embeddings["</e>"] = np.random.uniform(-0.01, 0.01, size=emb_dim)

    for emb in embedding:
        line = emb.strip().split()
        word = line[0]
        word_emb = np.array([float(_) for _ in line[1:]])
        embeddings[word] = word_emb

    vocab_list = list(embeddings.keys())
    word2id = {vocab_list[i]: i for i in range(len(vocab_list))}
    embedding_matrix = np.array(list(embeddings.values()))

    return embedding_matrix, word2id


if torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
else:
    device = torch.device("cpu")
idx2intent, intent2idx = lord_label_dict("../data/agis/intent_label.txt")
idx2slot, slot2idx = lord_label_dict("../data/agis/slot_label.txt")
embedding_file = open("../data/agis/emb_word.txt", "r", encoding="utf-8")
embeddings = [emb.strip() for emb in embedding_file]
embedding_word, vocab = process_emb(embeddings, emb_dim=300)

model = torch.load('agis_model.bin', map_location=device)
model.eval()


def question_deal(sent):
    print(sent)
    # 识别意图及槽位
    # slot, intent = text.split(',')

    pred_intents = []
    pred_slots = []
    # sent = '玉米瘤黑穗病主要危害玉米的哪些部位？'
    inputs = [[vocab[word] for word in list(sent)] + [vocab["<pad>"]] * (32 - len(sent))]
    char_lists = []
    masks = [[1] * len(sent) + [0] * (32 - len(sent))]

    if torch.cuda.is_available():
        inputs, char_lists, masks = torch.tensor(inputs).cuda(), torch.tensor(char_lists).cuda(), torch.tensor(
            masks).cuda()
    logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
    pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
    pred_intents.extend(pred_intent.cpu().numpy().tolist())

    for i in range(len(pred_slot)):
        pred = []
    for j in range(len(pred_slot[i])):
        pred.append(idx2slot[pred_slot[i][j].item()])
    # pred_slots.append(pred)

    # slots = parse_slot(pred, sent)

    # ['B-DIS', 'I-DIS', 'I-DIS', 'I-DIS', 'I-DIS', 'O', 'O', 'O', 'O', 'O']

    pred_intents = [idx2intent[intent] for intent in pred_intents]

    # if len(pred_intents) == 1:
    #     responce = pred_intents[0]
    # else:
    #     responce = "存在多个意图"

    print(pred)
    print(pred_intents)

if __name__ == '__main__':


    sent='玉米大斑病如何防治？'

    question_deal(sent)
