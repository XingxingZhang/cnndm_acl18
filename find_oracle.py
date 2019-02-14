import sys
import itertools
import gc
import math
import datetime

from PyRouge.Rouge.Rouge import Rouge
from Document import Document

rouge = Rouge(use_ngram_buf=True)

MAX_COMB_L = 5
MAX_COMB_NUM = 100000


def c_n_x(n, x):
    if x > (n >> 2):
        x = n - x
    res = 1
    for i in range(n, n - x, -1):
        res *= i
    for i in range(x, 0, -1):
        res = res // i
    return res


def solve_one(document):
    if document.doc_len == 0 or document.summary_len == 0:
        return None, 0
    sentence_bigram_recall = [0] * document.doc_len
    for idx, sent in enumerate(document.doc_sents):
        scores = rouge.compute_rouge([document.summary_sents], [sent])
        recall = scores['rouge-2']['r'][0]
        # print('recal is', recall)
        sentence_bigram_recall[idx] = recall
    candidates = []
    for idx, recall in enumerate(sentence_bigram_recall):
        if recall > 0:
            candidates.append(idx)
    if len(candidates) == 0:
        print('candidate is zero!!!')
    all_best_l = 1
    all_best_score = 0
    all_best_comb = None
    for l in range(1, len(candidates)+1):
        if l > MAX_COMB_L:
            print('Exceed MAX_COMB_L')
            break
        comb_num = c_n_x(len(candidates), l)
        if math.isnan(comb_num) or math.isinf(comb_num) or comb_num > MAX_COMB_NUM:
            print('Exceed MAX_COMB_NUM')
            break
        combs = itertools.combinations(candidates, l)
        l_best_score = 0
        l_best_choice = None
        for comb in combs:
            c_string = [document.doc_sents[idx] for idx in comb]
            rouge_scores = rouge.compute_rouge([document.summary_sents], [c_string])
            rouge_bigram_f1 = rouge_scores['rouge-2']['f'][0]
            if rouge_bigram_f1 > l_best_score:
                l_best_score = rouge_bigram_f1
                l_best_choice = comb
        if l_best_score > all_best_score:
            all_best_l = l
            all_best_score = l_best_score
            all_best_comb = l_best_choice
        else:
            if l > all_best_l:
                break
    return all_best_comb, all_best_score


def solve_one_rouge1(document):
    if document.doc_len == 0 or document.summary_len == 0:
        return None, 0
    sentence_bigram_recall = [0] * document.doc_len
    for idx, sent in enumerate(document.doc_sents):
        scores = rouge.compute_rouge([document.summary_sents], [sent])
        recall = scores['rouge-1']['r'][0]
        sentence_bigram_recall[idx] = recall
    candidates = []
    for idx, recall in enumerate(sentence_bigram_recall):
        if recall > 0:
            candidates.append(idx)
    if len(candidates) == 0:
        print('candidate is zero!!!')
    all_best_l = 1
    all_best_score = 0
    all_best_comb = None
    for l in range(1, len(candidates)+1):
        if l > MAX_COMB_L:
            print('Exceed MAX_COMB_L')
            break
        comb_num = c_n_x(len(candidates), l)
        if math.isnan(comb_num) or math.isinf(comb_num) or comb_num > MAX_COMB_NUM:
            print('Exceed MAX_COMB_NUM')
            break
        combs = itertools.combinations(candidates, l)
        l_best_score = 0
        l_best_choice = None
        for comb in combs:
            c_string = [document.doc_sents[idx] for idx in comb]
            rouge_scores = rouge.compute_rouge([document.summary_sents], [c_string])
            rouge_bigram_f1 = rouge_scores['rouge-1']['f'][0]
            if rouge_bigram_f1 > l_best_score:
                l_best_score = rouge_bigram_f1
                l_best_choice = comb
        if l_best_score > all_best_score:
            all_best_l = l
            all_best_score = l_best_score
            all_best_comb = l_best_choice
        else:
            if l > all_best_l:
                break
    return all_best_comb, all_best_score


def solve(documents, output_file):
    def label_comb(comb, lbl):
        assert len(comb) == 2
        comb_ = list(comb)
        comb_.append(lbl)
        return tuple(comb_)

    writer = open(output_file, 'w', encoding='utf-8', buffering=1)
    for idx, doc in enumerate(documents):
        print('doc', idx)
        if idx % 50 == 0:
            print(datetime.datetime.now())
            rouge.ngram_buf = {}
            gc.collect()
        comb = solve_one(doc)
        comb = label_comb(comb, 'rouge-2')
        if comb[0] is None:
            comb = solve_one_rouge1(doc)
            comb = label_comb(comb, 'rouge-1')
        writer.write('{0}\t{1}\t{2}\n'.format(comb[0], comb[1], comb[2]))
    writer.close()


def load_data(src_file, tgt_file):
    docs = []
    with open(src_file, 'r', encoding='utf-8') as src_reader, \
            open(tgt_file, 'r', encoding='utf-8') as tgt_reader:
        for src_line, tgt_line in zip(src_reader, tgt_reader):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line == "" or tgt_line == "":
                docs.append(None)
                continue
            src_sents = src_line.split('##SENT##')
            tgt_sents = tgt_line.strip().split('##SENT##')
            docs.append(Document(src_sents, tgt_sents))
    return docs


def main(src_file, tgt_file, outfile_name):
    docs = load_data(src_file, tgt_file)
    solve(docs, outfile_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
