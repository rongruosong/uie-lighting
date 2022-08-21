# coding=utf-8
import os
import argparse

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers import (ErnieForPretraining, ErniePretrainedModel,
                                    ErnieTokenizer, AutoTokenizer)
from transformers import BertConfig, BertTokenizer
import torch
from ernie import UIETorch
from model import UIE

def change_paddle_key():
    paddle_state_dict = {}

    # embedding
    paddle_state_dict['ernie.embeddings.word_embeddings.weight'] = 'encoder.embeddings.word_embeddings.weight'
    paddle_state_dict['ernie.embeddings.position_embeddings.weight'] = 'encoder.embeddings.position_embeddings.weight'
    paddle_state_dict['ernie.embeddings.token_type_embeddings.weight'] = 'encoder.embeddings.token_type_embeddings.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.weight'] = 'encoder.embeddings.layer_norm.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.bias'] = 'encoder.embeddings.layer_norm.bias'
    paddle_state_dict['ernie.embeddings.task_type_embeddings.weight'] = 'encoder.embeddings.task_type_embeddings.weight'

    # encoder
    for i in range(12):
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.weight'.format(i)] = 'encoder.encoder.layers.{}.self_attn.q_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.bias'.format(i)] = 'encoder.encoder.layers.{}.self_attn.q_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.weight'.format(i)] = 'encoder.encoder.layers.{}.self_attn.k_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.bias'.format(i)] = 'encoder.encoder.layers.{}.self_attn.k_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.weight'.format(i)] = 'encoder.encoder.layers.{}.self_attn.v_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.bias'.format(i)] = 'encoder.encoder.layers.{}.self_attn.v_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.weight'.format(i)] = 'encoder.encoder.layers.{}.self_attn.out_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.bias'.format(i)] = 'encoder.encoder.layers.{}.self_attn.out_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.weight'.format(i)] = 'encoder.encoder.layers.{}.norm1.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.bias'.format(i)] = 'encoder.encoder.layers.{}.norm1.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.weight'.format(i)] = 'encoder.encoder.layers.{}.linear1.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.bias'.format(i)] = 'encoder.encoder.layers.{}.linear1.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.weight'.format(i)] = 'encoder.encoder.layers.{}.linear2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.bias'.format(i)] = 'encoder.encoder.layers.{}.linear2.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.weight'.format(i)] = 'encoder.encoder.layers.{}.norm2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.bias'.format(i)] = 'encoder.encoder.layers.{}.norm2.bias'.format(i) 

    paddle_state_dict['ernie.pooler.dense.weight'] = 'encoder.pooler.dense.weight'
    paddle_state_dict['ernie.pooler.dense.bias'] = 'encoder.pooler.dense.bias'
    paddle_state_dict['linear_start.weight'] = 'linear_start.weight'
    paddle_state_dict['linear_start.bias'] = 'linear_start.bias'
    paddle_state_dict['linear_end.weight'] = 'linear_end.weight'
    paddle_state_dict['linear_end.bias'] = 'linear_end.bias'

    return paddle_state_dict

def convert(model, pd_model_weight_path, save_path):
    # 加载paddle参数
    paddle_key_params = paddle.load(pd_model_weight_path)

    paddle_state_dict = change_paddle_key()
    state_dict = model.state_dict()
    for key in state_dict.keys():
        
        if key in paddle_state_dict.keys():
            param = paddle_key_params[paddle_state_dict[key]]
            if 'weight' in key and 'LayerNorm' not in key and 'embeddings' not in key and 'decoder' not in key:
                param = param.transpose((1, 0))
            state_dict[key] = torch.from_numpy(param.numpy())
        else:
            print(key)
    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), save_path)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./checkpoint/model_best', help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export', help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()

if __name__ == '__main__':
    pd_model_weight_path = '/root/autodl-nas/uie-base/model_state.pdparams'
    config_dir = '/root/autodl-nas/uie-base/model_config.json'
    config = BertConfig.from_json_file(config_dir)
    model = UIETorch(config, True)
    for k in model.state_dict().keys():
        print(k, " ", model.state_dict()[k].shape)

    save_path = '/root/autodl-nas/uie-base/pytorch_model.bin'
    # convert(model, pd_model_weight_path, save_path)

    # 验证参数转换后， torch的结果是否与paddle保持一致
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-nas/uie-base/')
    input = tokenizer.encode('百度百科')
    print(input)

    # torch
    model.load_state_dict(torch.load('/root/autodl-nas/uie-base/pytorch_model.bin'), strict=False)
    model.eval()
    ids = torch.LongTensor(np.expand_dims(input, 0))
    print(model(ids, None, None, None))

    # paddle
    tokenizer = ErnieTokenizer.from_pretrained('/root/autodl-nas/uie-base/')
    input = tokenizer.encode('百度百科')
    ids = paddle.to_tensor(np.expand_dims(input['input_ids'], 0))
    model = UIE.from_pretrained('/root/autodl-nas/uie-base/')
    model.eval()
    print(model(ids))
