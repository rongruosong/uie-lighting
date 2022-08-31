# coding=utf-8
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, FileOpener
from typing import IO, Tuple, Dict, Iterator, List, Any
import json
from transformers import BertTokenizerFast
import torch

@functional_datapipe("parse_line_json_files")
class LineJsonParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    r"""
    加载解析分行的json文件(functional name: ``parse_line_json_files``).
    Args:
        source_datapipe: a DataPipe with tuples of file name and JSON data stream
        kwargs: keyword arguments that will be passed through to ``json.loads``
    """

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, IO]], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, IO]] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Dict]:
        for file_name, stream in self.source_datapipe:
            for line in stream:
                yield json.loads(line, **self.kwargs)


@functional_datapipe("truncation_line")
class LineTruncationIterDataPipe(IterDataPipe):
    r"""
    处理每行数据截取最大长度
    """
    def __init__(self, source_datapipe: IterDataPipe[Dict[str, Any]], max_seq_len: int):
        self.source_datapipe = source_datapipe
        self.max_seq_len = max_seq_len
    
    def __iter__(self):
        for json_line in self.source_datapipe:
            content = json_line['content'].strip()
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if self.max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = self.max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len =self.max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


@functional_datapipe('convert_example')
class ConvertIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[Dict[str, Any]], tokenizer: BertTokenizerFast, max_seq_len: int):
        self.source_datapipe = source_datapipe
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for example in self.source_datapipe:
            encoded_inputs = self.tokenizer.encode_plus(text=example["prompt"],
                                                        text_pair=example["content"],
                                                        padding='max_length',
                                                        truncation=True,
                                                        max_length=self.max_seq_len,
                                                        return_offsets_mapping=True)
            encoded_inputs = encoded_inputs
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
            start_ids = [0 for x in range(self.max_seq_len)]
            end_ids = [0 for x in range(self.max_seq_len)]
            for item in example["result_list"]:
                start = self.map_offset(item["start"] + bias, offset_mapping)
                end = self.map_offset(item["end"] - 1 + bias, offset_mapping)
                start_ids[start] = 1.0
                end_ids[end] = 1.0

            tokenized_output = [
                encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
                encoded_inputs["attention_mask"],sum(encoded_inputs["attention_mask"]),
                start_ids, end_ids
            ]
            tokenized_output = [torch.tensor(x, dtype=torch.int64) for x in tokenized_output]
            yield tuple(tokenized_output)

    def map_offset(self, ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1


def collate_fn(batch):
    input_ids, token_type_ids, attention_mask, lengths, start_ids, end_ids = map(torch.stack, zip(*batch))
    max_len = max(lengths).item()
    output = [input_ids, token_type_ids, attention_mask, start_ids, end_ids]
    return tuple([x[:, :max_len] for x in output])


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('/root/autodl-nas/uie-base/')
    datapipe1 = IterableWrapper(['/root/train.txt'])
    datapipe2 = FileOpener(datapipe1, mode="b")
    datapipe3 = datapipe2.parse_line_json_files()
    datapipe4 = datapipe3.truncation_line(max_seq_len=50)
    datapipe5 = datapipe4.convert_example(tokenizer=tokenizer, max_seq_len=50).batch(3)
    datapipe6 = datapipe5.collate(collate_fn)
    print(next(iter(datapipe6)))