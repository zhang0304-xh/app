import torch
import numpy as np
from py2neo import Graph
import json
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

idx2intent, intent2idx = lord_label_dict("data/agis/intent_label.txt")
idx2slot, slot2idx = lord_label_dict("data/agis/slot_label.txt")
embedding_file = open("data/agis/emb_word.txt", "r", encoding="utf-8")
embeddings = [emb.strip() for emb in embedding_file]
embedding_word, vocab = process_emb(embeddings, emb_dim=300)

model = torch.load('model/agis_model.bin', map_location=device)
model.eval()

adict_light = {"1": 0, "2": 1, "3": 2, "4": 3}  # 字典
adict_growth_perid = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

adict_nitrogen = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

slot2word = {
    'B-CRO': '作物',
    'B-CLA': '其他',
    'B-DIS': '病害',
    'B-DRUG': '药剂',
    'B-PART': '部位',
    'B-PER': '时期',
    'B-PET': '虫害',
    'B-REA': '病原',
    'B-STRAINS': '品种',
    'B-SYM': '症状',
    'B-WEE': '草害'
}

def parse_slot(slot_labels, text):
    # print(text)
    # print(slot_labels)
    keywords = {}
    entity = []
    entity_type = ''
    for i, slot in enumerate(slot_labels):
        if slot == 'O':
            if entity:
                if slot2word[entity_type] not in keywords.keys():
                    keywords[slot2word[entity_type]] = [''.join(entity)]
                else:
                    keywords[slot2word[entity_type]].append(''.join(entity))
                entity = []
                entity_type = ''

        if slot.startswith('B-'):
            if entity:
                if slot2word[entity_type] not in keywords.keys():
                    keywords[slot2word[entity_type]] = [''.join(entity)]
                else:
                    keywords[slot2word[entity_type]].append(''.join(entity))
                entity = []
                entity_type = ''
            else:
                entity.append(text[i])
                entity_type = slot
        if slot.startswith('I-'):
            if not entity:
                entity.append(text[i])
                entity_type = 'B-' + slot.split('-')[-1]
            else:
                entity.append(text[i])

    if entity:
        if slot2word[entity_type] not in keywords.keys():
            keywords[slot2word[entity_type]] = [''.join(entity)]
        else:
            keywords[slot2word[entity_type]].append(''.join(entity))
    return keywords

def question_deal(sent):
    # try:

        graph = Graph("http://localhost:7474", auth=("neo4j", "12345678"))
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

        slots = parse_slot(pred, sent)


        # ['B-DIS', 'I-DIS', 'I-DIS', 'I-DIS', 'I-DIS', 'O', 'O', 'O', 'O', 'O']
        pred_intents = [idx2intent[intent] for intent in pred_intents]

        # if len(pred_intents) == 1:
        #     responce = pred_intents[0]
        # else:
        #     responce = "存在多个意图"

        cypher = "MATCH(n1: {0})-[r: {1}]->(n2) where  n1.name= '{2}' return n2.name as content"  # 查询模板

        keywords = list(slots.values())[0][0]

        #print(keywords)
        cypher = cypher.format(list(slots.keys())[0], pred_intents[0], keywords)
        print(cypher)
        #匹配查出的关键词
        #sas = f"MATCH (n:{keywords[0]})-[r:{keywords[1]}]->(n2)  return n2"
        data = graph.run(cypher).data()

        #json_data = json.dumps(data, ensure_ascii=False)

        return data#json_data
        # responce = []
        # for a in data:
        #     for tk, tv in a.items():
        #         nodes = tv.nodes
        #         # _node = Node(nodes[0])
        #         for n in nodes:
        #             obj_properties = {}
        #             for k, v in n.items():
        #                 obj_properties[k] = v
        #
        #             print(obj_properties)
        #             responce.append(obj_properties['name'])
        #
        # if not responce:
        #     return '抱歉，您的问题暂未收录'
        # else:
        #
        #     if len(responce) > 2030:
        #         responce = responce[:2030] + '...'
        #
        #     return ','.join(responce)
    # except:

            # return '您的问题暂未收录'
        # return cypher



#
# if __name__ == '__main__':
#
#     sent='玉米大斑病如何防治？'
#
#     print(question_deal(sent))
