import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import argparse
import random
import pickle
from tqdm import tqdm
import numpy as np
import re

from generator import Generator

import json
from collections import defaultdict
from utils import *

merged_relations = [
        'belongs to the category of',
        'is a category',
        'may cause',
        'is a subtype of',
        'is a risk factor of',
        'is associated with',
        'may contraindicate',
        'interacts with',
        'belongs to the drug family of',
        'belongs to drug super-family',
        'is a vector for',
        'may be allelic with',
        'see also',
        'is an ingradient of',
        'may treat'
    ]

relas_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 5, "10": 6, "12": 7, "16": 8, "17": 9, "18": 10,
                "20": 11, "26": 12, "30": 13, "233": 14}


def load_ddb():            
    with open(f'./data/ddb/ddb_names.json') as f:
        all_names = json.load(f)
    with open(f'./data/ddb/ddb_relas.json') as f:
        all_relas = json.load(f)
    relas_lst = []
    for key, val in all_relas.items():
        relas_lst.append(val)
        
    ddb_ptr_to_preferred_name = {}
    ddb_ptr_to_name = defaultdict(list)
    ddb_name_to_ptr = {}
    for key, val in all_names.items():
        item_name = key
        item_ptr = val[0]
        item_preferred = val[1]
        if item_preferred == "1":
            ddb_ptr_to_preferred_name[item_ptr] = item_name
        ddb_name_to_ptr[item_name] = item_ptr
        ddb_ptr_to_name[item_ptr].append(item_name)
        
    return (relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name)

        
        
if __name__ == '__main__':
    # random.seed(42)
    args = parse_args()
    seed = args.seed
    set_seed(seed)
    print(args)
    
    model = Generator("./models/7B")
    sys_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: %s\nASSISTANT:"

    contriever = AutoModel.from_pretrained("facebook/contriever-msmarco").to(args.device)
    contriever_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    # bert_tokenizer  = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    # bert_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(args.device)
    print("Finished loading model")
    
    relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()
    all_facts = []
    for relation in relas_lst: 
            head_ent = ddb_ptr_to_preferred_name[relation[0]]
            rel = merged_relations[relas_dict[relation[2]]]
            tail_ent = ddb_ptr_to_preferred_name[relation[1]]
            all_facts.append(head_ent+' '+rel+' '+tail_ent)
    
    if args.loademb:
        all_embs = torch.load("./data/DDB_embs.pt", map_location=args.device)
    else:
        all_embs = get_sent_embeddings(all_facts, contriever, contriever_tokenizer, BSZ=1024)

    
    question_list, ans_ground_list, ans_can_list, q_ent_list, ans_ent_list =  load_question_ent(args.path)
    lines = load_dataset("./data/test.sample.kg.json")
    medicalKG_head_dict = load_triplets_dict("./data/ddb_head_kg.pkl")
    medicalKG_tail_dict = load_triplets_dict("./data/ddb_tail_kg.pkl")
    
    print("Finished embedding")
    
    total_ques = 0
    cor = 0
    for i, question in enumerate(question_list):
        
        if i % 10 == 0:
            print(f"Finised on {i}-th case, acc: {cor/(i+1e-12)}")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        ans_can = ans_can_list[i]
        true_ans = ans_ground_list[i]
        
        ans_fact_list =[]
        ans_emb_list = []
        for ans in ans_can:
            select_fact, selected_emb = retr_fact(ans, all_embs, all_facts, args.fact_number*2, contriever, contriever_tokenizer)
            ans_fact_list += select_fact
            ans_emb_list += selected_emb 
        
        ans_emb = torch.stack(ans_emb_list)
        facts = ans_fact_list
        if len(facts) > args.fact_number:
            inputs = question + ' ' + ', '.join(ans_can)
            select_fact, _ = retr_fact(inputs, ans_emb, facts, args.fact_number, contriever, contriever_tokenizer)
        else:
            select_fact = facts
        select_fact = ', '.join(select_fact)
        print("*******select_fact", select_fact)
        questions = build_big_template(question, select_fact, ans_can)
        
 
        if args.model == 'vicuna':
            ans = model.generate([sys_template % questions])[0]
        else:
            pass
        print('ans', ans)
        
        simple_truth_ans = re.sub(r"[^a-zA-Z ]+", '', true_ans).lower()
        simple_ans = re.sub(r"[^a-zA-Z ]+", '', ans).lower()
        print(f"ground_truth: {simple_truth_ans}, ans: {simple_ans}")
        print("=========================================================")
        if simple_truth_ans in simple_ans:
            cor += 1
    print(cor/(i+1))
    
    