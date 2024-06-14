# -- coding:utf-8 --
import torch
from transformers import BertTokenizer, BertConfig
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from LawKeywordBert.models.bert_for_ner import BertSpanForNer
from LawKeywordBert.processors.ner_span import ner_processors as processors
from LawKeywordBert.processors.ner_span import collate_fn, InputFeature
from LawKeywordBert.processors.utils_ner import bert_extract_item
from torch.utils.data import DataLoader, SequentialSampler
import argparse
from utils import get_cls_token
import jieba
import jieba.analyse


def get_parse():
    parser = argparse.ArgumentParser('KeywordExtraction')
    parser.add_argument("--task_name", type=str, default='LKR')
    parser.add_argument("--data_dir", type=str, default='bert_ner/datasets/LKR')
    parser.add_argument("--train_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    return parser


def diverse_keywords(
        word_embedding,
        words,
        top_n: int = 10,
        diverse_method: str = 'kmeans'
):
    word_embedding = word_embedding.cpu().detach().numpy()

    if diverse_method == 'kmeans':
        kmeans_model = KMeans(n_clusters=min(len(word_embedding), top_n), n_init=10, random_state=42)
        cluster = kmeans_model.fit_predict(word_embedding)

        cluster_centers = kmeans_model.cluster_centers_

        results = []
        for c_id, center in enumerate(cluster_centers):
            min_distance = np.inf
            cluster_words_embedding = word_embedding[cluster == c_id]  # the embedding of the cluster words
            indices_in_cluster = np.where(cluster == c_id)[0]  # the indices of the cluster words
            cluster_words = [words[iic] for iic in indices_in_cluster]  # the word of the cluster
            candidate = None
            for word_id in range(len(cluster_words)):
                cosine_similarity = (np.dot(center, cluster_words_embedding[word_id]) /
                                     (np.linalg.norm(center) * np.linalg.norm(cluster_words_embedding[word_id])))
                if cosine_similarity < min_distance:
                    min_distance = cosine_similarity
                    candidate = cluster_words[word_id]
            if candidate:
                results.append(candidate)
        return results

    else:
        raise ValueError('The {} is not supported.'.format(diverse_method))


def get_keyword(
        case: str,
        keywords: list,
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        k: int = 20,
        n: int = 10,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        diverse_method: str = 'msd'
):
    result = []
    candidate_word_embedding = []

    if len(keywords) == 0:
        return []

    word_tensor_db = []
    for word in keywords:
        word_cls = get_cls_token(word, tokenizer, model, config, device, for_word=True)
        word_tensor_db.append(word_cls)

    word_tensor_db = torch.stack(word_tensor_db).squeeze(1)

    doc_embedding = get_cls_token(case, tokenizer, model, config, device)
    doc_embedding = doc_embedding.expand(word_tensor_db.shape[0], 768)

    cosine_sim = F.cosine_similarity(word_tensor_db, doc_embedding, dim=-1)
    top_values, top_indices = torch.topk(cosine_sim, k=min(len(keywords), k))
    top_indices = top_indices.tolist()
    for value, idx in zip(top_values, top_indices):
        result.append((keywords[idx], round(value.item(), 4)))
        candidate_word_embedding.append(word_tensor_db[idx])

    candidate_words = [word[0] for word in result]
    keyword_result = diverse_keywords(
        torch.stack(candidate_word_embedding),
        candidate_words,
        top_n=n,
        diverse_method=diverse_method,
    )
    # print(keyword_result)
    return keyword_result


def sentence_based_truncation(
        case: str
) -> list:
    case_segment = []

    while len(case) >= 510:
        segment_pun = ['。', '；', ';', '，', ',']
        period_index = -1
        for pun in segment_pun:
            period_index = case[:510].rfind(pun)
            if period_index != -1:
                break

        case_segment.append(case[:period_index + 1])
        case = case[period_index + 1:]

    if len(case) >= 10:
        case_segment.append(case)

    return case_segment


def keyword_prediction(
        tokenizer: BertTokenizer,
        model: BertSpanForNer,
        config: BertConfig,
        task_name,
        examples: str,
        k: int = 20,
        n: int = 10,
        train_max_seq_length: int = 512,
        eval_max_seq_length: int = 512,
        device: torch.device = torch.device('cpu'),
        method: str = 'ner',
        diverse_method: str = 'kmeans',
) -> list:
    if method == 'ner':
        # split the examples so that each new examples are shorter than 510
        case_segment = sentence_based_truncation(examples)

        test_dataset = load_and_cache_examples(task_name, case_segment, tokenizer, 'test', train_max_seq_length,
                                               eval_max_seq_length)
        # print(len(test_dataset))

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

        processor = processors[task_name]()
        label_list = processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}

        results = []
        for step, batch in enumerate(test_dataloader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None,
                          "token_type_ids": batch[2]}
                outputs = model(**inputs)
            start_logits, end_logits = outputs[:2]
            R = bert_extract_item(start_logits, end_logits)
            if R:
                label_entities = [[id2label[x[0]], x[1], x[2]] for x in R]
            else:
                label_entities = []
            json_d = {'id': step, 'entities': label_entities}
            results.append(json_d)

        all_keywords = []
        for res, example in zip(results, case_segment):
            tag_dict = {'K': []}
            for entity in res['entities']:
                tag, start, end = entity
                if tag in tag_dict:
                    tag_dict[tag].append(example[start: end + 1])

            keywords = tag_dict['K']
            all_keywords.extend(keywords)

    elif method == 'srm':
        thu_ocl = []
        with open('data/THUOCL_law.txt', 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                word = line.split('\t')[0]
                thu_ocl.append(word.strip())
                line = f.readline()

        all_keywords = []
        for word in thu_ocl:
            if word in examples:
                all_keywords.append(word)

    elif method == 'tfidf':
        all_keywords = jieba.analyse.extract_tags(examples, withWeight=True, topK=10, allowPOS=('n', 'v', 'vd', 'vn', 'a', 'ad', 'an', 'd', 'm', 'q'))
        all_keywords = [k[0] for k in all_keywords]

    elif method == 'textrank':
        all_keywords = jieba.analyse.textrank(examples, topK=20, withWeight=False, allowPOS=('n', 'v', 'vd', 'vn', 'a', 'ad', 'an', 'd', 'm', 'q'))
        all_keywords = [k[0] for k in all_keywords]

    else:
        raise ValueError('{} method is not supported.'.format(method))

    if all_keywords:
        all_keywords = get_keyword(examples, list(set(all_keywords)), tokenizer, model,
                                   config=config, k=k, n=n, device=device,
                                   diverse_method=diverse_method)
    else:
        all_keywords = ['无']
    # print(all_keywords)
    return all_keywords


if __name__ == '__main__':
    pass
