import torch
import json
import argparse
import random
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="In Context Learning for pretrained GPTs")
    parser.add_argument('--device', type=str, default = "cuda")
    parser.add_argument('--model',type=str, default='vicuna')
    parser.add_argument('--path',type=str, default='./data/medqa_usmle/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--prompt_number', type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--fact_number', type=int, default=8)
    parser.add_argument('--loademb', type=bool, default=True)
    parser.add_argument('--summarize', type=bool, default=False)
    parser.add_argument('--source', type=str, default='wiki')
    
    args = parser.parse_args()
    return args  

args = parse_args()


def retrieve_facts(query, fact_embs, contriever, tok, fact_number=1):
        inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(contriever.device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            query_emb = mean_pooling(outputs[0], inputs['attention_mask'])
        sim = (query_emb @ fact_embs.T)[0]
        knn = sim.topk(fact_number, largest=True)
        return knn.indices

    
def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
        all_embs = []
        for i in range(0, len(sents), BSZ):
            sent_batch = sents[i:i+BSZ]
            inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(contriever.device)
            with torch.no_grad():
                outputs = contriever(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            all_embs.append(embeddings.cpu())
        all_embs = torch.vstack(all_embs)
        return all_embs
    
def retr_fact(input, embs, new_facts_set, fact_number, contriever, contriever_tokenizer):
    
    fact_ids = retrieve_facts(input, embs, contriever, contriever_tokenizer, fact_number)
    selected_fact = [new_facts_set[fact_id] for fact_id in fact_ids]
    selected_emb = embs[fact_ids]  
    
    return selected_fact, selected_emb
    #return selected_fact 

def load_triplets_dict(filename):
    with open(filename, 'rb') as file:
        triplets_dict = pickle.load(file)
    return triplets_dict


def build_question(lines):
    
    questions = []
    for d in lines:
        
        question = d["question"]
        questions.append(question)
    
    return questions

def build_rethink_template(question, ans):
    
    #whole_rethink_template =  'Is this answer: ' + ans + ' correct for a medical question: ' + question + '  Please answer Yes or No.' 
    #whole_rethink_template =  'Given a medical question: ' + question + ', is the statment: ' +ans+' correct? Please answer me only in Yes or No.' 
    whole_rethink_template =  'Given a medical question: ' + question + ', is the answer: ' +ans+' correct? Please answer me in Yes or No with an explanation' 
    
    return whole_rethink_template

def build_template(question, ans_can_list):

    #whole_question_template = 'You are a medical assistant and your task is to output medical answer to a given quesiton. Given question: ' + question + ' Please output only one answer from the following choices: '
    whole_question_template =  question + ' Which of the following answer is true: '
    for i , ans_can in enumerate(ans_can_list):
        #whole_question_template = whole_question_template + f'{i}. '+ ans_can + '. '
        whole_question_template = whole_question_template + ans_can + '. '
    whole_question_template  = whole_question_template + 'You can only output the predicted label in exact words. No other words should be included.'
    
    return whole_question_template


def build_big_template(question, facts, ans_can_list):

    #whole_question_template = 'You are a medical assistant and your task is to output medical answer to a given quesiton. Here are facts you may find useful: '+ facts + '. Given question: ' + question + ' Please output only one answer from the following choices: '
    whole_question_template = 'Here are some medical facts: '+ facts + '. Given question: ' + question + ' Which of the following answer is true: '
    for i , ans_can in enumerate(ans_can_list):
        #whole_question_template = whole_question_template + f'{i}. '+ ans_can + '. '
        whole_question_template = whole_question_template + ans_can + '. '
    whole_question_template  = whole_question_template + 'You can only output the predicted label in exact words. No other words should be included.'
    
    return whole_question_template

def build_template_question(lines):
    
    questions = []
    for d in lines:
        question = d["question"]
        ans_can_list =  d["choices"]
        whole_question_template = 'You are a medical assistant and your task is to give medical answer a given quesiton. Given question: ' + question + ' Please output only one answer from the following choices: '
        for ans_can in ans_can_list:
            whole_question_template = whole_question_template + ans_can["text"] + '. '
        whole_question_template  = whole_question_template + 'You should only output the predicted label in exact words. No other words should be included.'
        questions.append(whole_question_template)
    
    return questions

def build_knowledge_graph(ent_set, KG):
    
    KG_set = set()
    for ent in ent_set:
        if ent in KG:
            facts = KG[ent]
            if len(facts) > args.fact_number:
                facts = random.choices(facts, k = args.fact_number)
            facts = set(facts)
            KG_set = KG_set.union(facts)
        
    return KG_set
  
  
def load_dataset(path):
    
    with open(path, "r") as f:
        lines = json.load(f)
    
    return lines        
        
 
def load_question_ent(grounded_path):
    
    id_dict = {"A":0, "B":1, "C":2, "D":3}
    
    with open(grounded_path+'statement/test.statement.jsonl', 'r', encoding='utf-8') as fin:
        
        
        ans_ground_id_list = []
        for line in fin:
            dic = json.loads(line)
            true_ans_id  = dic["answerKey"]  # in this step, we extract A, B, C, D
            ans_ground_id_list.append(true_ans_id)
        
    
    with open(grounded_path+'grounded/test.grounded.jsonl', 'r', encoding='utf-8') as fin:
        
        grouped_lines = []
        current_group = []
        question_list = []
        ans_ground_list = []
        ans_can_list = []
        q_ent_list = []
        ans_ent_list = []
        

        for line in fin:
            current_group.append(line)  # Remove leading/trailing whitespace
            if len(current_group) == 4:
                grouped_lines.append(current_group)
                current_group = []

        for i, grouped_line in enumerate(grouped_lines):
            #print(grouped_line)
            
            dic = json.loads(grouped_line[0])
            question = dic['sent']
            question_list.append(question)
            
            q_ent = set(c for c in dic['qc_names'])
            if not q_ent:
                q_ent = {""} 
            
            q_ent_list.append(q_ent)
            
            ans_set= set()
            line_ans_list = []
            for line in grouped_line:
                
                dic = json.loads(line)
                ground_ans = dic['ans']
                line_ans_list.append(ground_ans)
                
                ans_ent = set(c for c in dic['ac_names']) # for one answer, the ids can be multiple
                #print("ans_ent", ans_ent)
                if not ans_ent:
                    ans_ent = {''}
                ans_set = ans_set.union(ans_ent)
                #print("ans_set", ans_set)
            
            correct_ans_id =  ans_ground_id_list[i] # return A,B,C,D
            correct_ans_index = id_dict[correct_ans_id] #return 0,1,2,3
            true_ans = line_ans_list[correct_ans_index] #return real answer in string
            
            
            ans_ground_list.append(true_ans)
            ans_can_list.append(line_ans_list)
            ans_ent_list.append(ans_set)
        

            
    return question_list, ans_ground_list, ans_can_list, q_ent_list, ans_ent_list
                   
    
    
    