import os
import gc
import time
import json
import pickle as pkl
import functools

#import deepspeed
import tqdm
import numpy as np
import torch as tc
import transformers as trf


LAYER_ID = "{:LID:}"
WEIGHT_TYPE = "{:WT:}"
LLAMA_ATTN_PATTERN = "model.layers.{LID}.self_attn.{WT}_proj.weight"
LLAMA_NORM_PATTERN = "model.layers.{LID}.input_layernorm.weight"
LLAMA_PROJ_PATTERN = "model.layers.{LID}.mlp.{WT}_proj.weight"
GPT2_PATTERN = "transformer.h.{:LID:}.attn.c_{:WT:}.weight"

def zero_grad(*obj):
    if len(obj) > 1:
        for subobj in obj:
            zero_grad(subobj)
    elif hasattr(obj[0], "parameters"):
        for subobj in obj[0].parameters():
            zero_grad(subobj)
    elif obj[0].grad is not None:
        obj[0].grad.data.zero_()



def format_llama_weight(layer, wtype):
    assert wtype in {"q", "k", "v", "o", "gate", "down", "up", "norm"}
    if wtype == "norm":
        pattern = "input_layernorm"
    elif len(wtype) == 1:
        pattern = "self_attn.%s_proj" % wtype
    else:
        pattern = "mlp.%s_proj" % wtype
    return "model.layers.%s.%s.weight" % (layer, pattern)


class Generator:
    def __init__(self, model, device="cuda:0", **params):
        super().__init__()
        self._name = model
        self._device = device
        self._params = {}
        self.parameters = params
        self.build()      

    def build(self):
        print("Initializing LLM: %s" % self._name)
        maps = "cpu" if self._device == "cpu" else "auto"
        self._tokenizer = trf.AutoTokenizer.from_pretrained(self._name, use_fast=False, padding_side="left", cache_dir="./cache")
        self._model = trf.AutoModelForCausalLM.from_pretrained(self._name, cache_dir="./cache", device_map=maps).float()
        self._out_embed = self._model.get_output_embeddings().weight.data.detach()
        self._embed = self._model.get_input_embeddings()
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id  
        self._config = self._model.config
        self._headsize = self._config.hidden_size // self._config.num_attention_heads

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._params.copy()

    @parameters.setter
    def parameters(self, params):
        for key, val in self._params:
            if key not in params:
                params[key] = val
        self._params = {"min_length": params.get("minlen", 1),
                        "max_length": params.get("maxlen", 300),
                        "temperature": params.get("temperature", 0.0),
                        "top_p": params.get("top_p", 0.1),
                        "num_return_sequences": params.get("ngen", 1),
                        "penalty_alpha": params.get("penalty", 0.),
                        "do_sample": False}      

    def get_inputs(self, texts):
        inputs = self._tokenizer(texts, padding=True, return_tensors="pt")
        for key in list(inputs.keys()):
            if key not in ["input_ids", "attention_mask"]:
                del inputs[key]
        return inputs
    
    def tokenize(self, text):
        return self._tokenizer.tokenize(text.strip())

    def prepare4generate(self, input_texts):
        if "pythia" in self._name:
            input_texts = ["<|prompter|>" + t + "<|assistant|>" for t in input_texts]
        inputs = self.get_inputs(input_texts) 
        batch_size, seq_len = inputs['input_ids'].shape
        
        inputs['attention_mask'] = tc.flip(inputs['attention_mask'], dims=[1])
        shifts = seq_len - inputs['attention_mask'].sum(dim=-1)
        for idx in range(batch_size):
            inputs['input_ids'][idx] = inputs['input_ids'][idx].roll(shifts[idx].item())

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()} | self._params
        if inputs['min_length'] is not None:
            inputs['min_length'] = inputs['min_length'] + seq_len
        if inputs['max_length'] is not None:
            inputs['max_length'] = min(self._model.config.max_position_embeddings,
                                       inputs['max_length'] + seq_len)
        return inputs, seq_len

    def generate(self, texts):
        with tc.no_grad():
            self._model.eval()
            inputs, seq_len = self.prepare4generate(texts)
            output_ids = self._model.generate(**inputs)[:, seq_len:]
            return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def forward(self, texts):
        inputs = self.get_inputs(texts)
        return self._model(**inputs)






