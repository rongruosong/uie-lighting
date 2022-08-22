# coding=utf-8
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import tqdm
import sys
import os
from collections import Counter, defaultdict

class UIEDataset(Dataset):
    def __init__(self, args, file_path, mode='train'):
        self.sample_size = 0
        self.sample_index = list()
        self.file = file_path
        self.tokenizer = args.tokenizer
        self.seq_length = args.seq_length
        self.label2id = args.label2id
        self.labels_num = len(self.label2id)
        self.mode = mode
        self._parse_dataset()

    def _parse_dataset(self):
        with open(self.file, mode='r', encoding='utf-8') as fin:
          self.sample_index.append(0)
          while True:
              line = fin.readline()
              if not line:
                  self.sample_index.pop()
                  break
              self.sample_index.append(fin.tell())
              self.sample_size += 1
    
    def convert_example(self, example, max_seq_len):
        """
        Adapted from paddleNLP
        example: {
            title
            prompt
            content
            result_list
        }
        """
        encoded_inputs = self.tokenizer(text=[example["prompt"]],
                                text_pair=[example["content"]],
                                truncation=True,
                                max_seq_len=max_seq_len,
                                pad_to_max_seq_len=True,
                                return_attention_mask=True,
                                return_position_ids=True,
                                return_dict=False,
                                return_offsets_mapping=True)
        encoded_inputs = encoded_inputs[0]
        offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
        bias = 0
        for index in range(1, len(offset_mapping)):
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias
        start_ids = [0 for x in range(max_seq_len)]
        end_ids = [0 for x in range(max_seq_len)]
        for item in example["result_list"]:
            start = self.map_offset(item["start"] + bias, offset_mapping)
            end = self.map_offset(item["end"] - 1 + bias, offset_mapping)
            start_ids[start] = 1.0
            end_ids[end] = 1.0

        tokenized_output = [
            encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
            encoded_inputs["position_ids"], encoded_inputs["attention_mask"],
            start_ids, end_ids
        ]
        tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
        return tuple(tokenized_output)


    def map_offset(self, ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

    def _convert(self, line):
        line = line.strip().split("\t")
        if self.mode in ['train', 'test']:
            labels = line[1]
            tgt = [self.label2id[l] for l in labels.split(" ")]

        text_a = line[0]
        src = self.tokenizer.convert_tokens_to_ids(
            text_a.split(' '))
        seg = [1] * len(src)

        length = len(src)
        if len(src) > self.seq_length:
            src = src[: self.seq_length]
            if self.mode in ['train', 'test']:
                tgt = tgt[: self.seq_length]
            seg = seg[: self.seq_length]
            length = self.seq_length
        while len(src) < self.seq_length:
            src.append(0)
            if self.mode in ['train', 'test']:
                tgt.append(self.labels_num)
            seg.append(0)
        src = torch.LongTensor(src)
        if self.mode in ['train', 'test']:
            tgt = torch.LongTensor(tgt)
        seg = torch.LongTensor(seg)
        length = torch.tensor(length, dtype=torch.int64)
        if self.mode == 'infer':
            return src, seg, length
        return src, tgt, seg, length

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        if index >= self.sample_size:
            raise IndexError
        index = self.sample_index[index]
        fin = open(self.file, encoding='utf-8')
        fin.seek(index)
        return self._convert(fin.readline())