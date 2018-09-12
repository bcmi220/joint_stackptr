__author__ = 'max'

import re
MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")


import sys
if sys.version_info[0] > 2:
    # py3k
    pass
else:
    # py2
    import codecs
    import warnings
    def open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
        if newline is not None:
            warnings.warn('newline is not supported in py2')
        if not closefd:
            warnings.warn('closefd is not supported in py2')
        if opener is not None:
            warnings.warn('opener is not supported in py2')
        return codecs.open(filename=file, mode=mode, encoding=encoding,
                    errors=errors, buffering=buffering)

import copy
def convert_to_conllu(eval_file, input_file, output_file, word_alphabet):

    def erase_multi_root(sent):
        childs = [[] for _ in range(len(sent)+1)]
        for jdx in range(len(sent)):
            head = int(sent[jdx][6])
            childs[head].append(jdx)
        
        if len(childs[0]) > 1: # multi root
            for jdx in range(1, len(childs[0])):
                sent[childs[0][jdx]][6] = str(childs[0][0]+1)

        return sent



    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = f.readlines()

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = f.readlines()

    eval_sentences = []
    sentence = []
    for line in eval_data:
        if len(line.strip())==0 or line.strip()[0]=='#':
            if len(sentence)>0:
                eval_sentences.append(sentence)
                sentence = []
        else:
            line = line.strip().split('\t')
            sentence.append(line)

    if len(sentence)>0:
        eval_sentences.append(sentence)
    
    input_sentences = []
    sentence = []
    for line in input_data:
        if len(line.strip())==0 or line.strip()[0]=='#':
            if len(sentence)>0:
                input_sentences.append(sentence)
                sentence = []
        else:
            line = line.strip().split('\t')
            sentence.append(line)

    if len(sentence)>0:
        input_sentences.append(sentence)

    # assert len(eval_sentences) == len(input_sentences)

    eval_sentences_map = {}
    for idx in range(len(eval_sentences)):
        line = ' '.join([item[1] for item in eval_sentences[idx]])
        eval_sentences_map[line] = idx

    total_sent = 0
    total_multi_root = 0
    raw_multi_root = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in range(len(input_sentences)):
            total_sent += 1
            eval_key = ' '.join([word_alphabet.get_instance(word_alphabet.get_index(DIGIT_RE.sub(b"0", item[1]))) for item in input_sentences[idx] if item[0].isdigit()])
            
            input_multi_root = 0
            for jdx in range(len(input_sentences[idx])):
                if input_sentences[idx][jdx][6] == '0':
                    input_multi_root += 1
            if input_multi_root != 1:
                raw_multi_root += 1

            if eval_sentences_map.get(eval_key) is not None:
                eval_idx = eval_sentences_map[eval_key]
                pred_multi_root = 0
                for jdx in range(len(eval_sentences[eval_idx])):
                    if eval_sentences[eval_idx][jdx][6] == '0':
                        pred_multi_root += 1

                if pred_multi_root != 1:
                    total_multi_root += 1
                
                if pred_multi_root == 1:
                    for jdx in range(len(input_sentences[idx])):
                        line = copy.deepcopy(input_sentences[idx][jdx])
                        if line[0].isdigit():
                            assert int(line[0])-1 >= 0 and int(line[0])-1 < len(eval_sentences[eval_idx])
                            line[6] = eval_sentences[eval_idx][int(line[0])-1][6]
                            line[7] = eval_sentences[eval_idx][int(line[0])-1][7]
                        f.write('\t'.join(line))
                        f.write('\n')
                    f.write('\n')
                elif input_multi_root == 1:
                    for jdx in range(len(input_sentences[idx])):
                        line = input_sentences[idx][jdx]
                        f.write('\t'.join(line))
                        f.write('\n')
                    f.write('\n')
                else:
                    sent = copy.deepcopy(eval_sentences[eval_idx])
                    sent = erase_multi_root(sent)
                    for line in sent:
                        f.write('\t'.join(line))
                        f.write('\n')
                    f.write('\n')
            else:
                if input_multi_root == 1:
                    for jdx in range(len(input_sentences[idx])):
                        line = copy.deepcopy(input_sentences[idx][jdx])
                        f.write('\t'.join(line))
                        f.write('\n')
                    f.write('\n')
                else:
                    sent = copy.deepcopy(input_sentences[idx])
                    sent = erase_multi_root(sent)
                    for line in sent:
                        f.write('\t'.join(line))
                        f.write('\n')
                    f.write('\n')

    print('total:%d multi-root:%d raw multi-root:%d'%(total_sent, total_multi_root,raw_multi_root))
