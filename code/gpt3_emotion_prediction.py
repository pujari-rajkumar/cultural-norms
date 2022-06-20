#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import datetime
import math
import json
import re


# #### Translations using hugging-face translation model

# In[2]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")


# In[3]:


def translate_hf(chinese_sent):
    batch = tokenizer([chinese_sent], return_tensors="pt")
    generated_ids = model.generate(**batch)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ### Loading data

# In[4]:


dir_path = '/homes/rpujari/scratch1_fortytwo/DARPA/'


# In[5]:


metadata = json.load(open(dir_path + 'mpdd/metadata.json'))
dialogue = json.load(open(dir_path + 'mpdd/dialogue.json'))


# In[6]:


field_rev = {}
pos_rev = {}
for key in metadata['field']:
    for reln in metadata['field'][key]:
        field_rev[reln] = key
for key in metadata['position']:
    for reln in metadata['position'][key]:
        pos_rev[reln] = key


# In[7]:


emotion_set = set()
for conv_id in dialogue:
    conv = dialogue[conv_id]
    for i, turn in enumerate(conv):
        emotion_set.add(turn['emotion'])
print(emotion_set)


# In[8]:


#selection dialogue instances where family members are disgusted/angry/surprised at an inferior family memeber 
sel_dialogues = set()
for conv_id in dialogue:
    conv = dialogue[conv_id]
    sel_dia = False
    for i, turn in enumerate(conv):
        if not sel_dia:
            if turn['emotion'] in ['disgust', 'anger', 'surprise']:
                sel = True
                for listener in turn['listener']:
                    reln = listener['relation']
                    if field_rev[reln] != 'family' or pos_rev[reln] != 'inferior':
                        sel = False
                if sel:
                    sel_dia = True
    if sel_dia:
        sel_dialogues.add(conv_id)
print(len(sel_dialogues))
print(len(dialogue))


# In[9]:


k = 118
for conv_id in list(sel_dialogues)[k:k+1]:
    print(conv_id)
    conv = dialogue[conv_id]
    for i, turn in enumerate(conv):
        print(turn['speaker'] + '(' + turn['emotion'] + ')' + ': ' +  turn['utterance'])
        print(translate_hf(turn['utterance']))
        if turn['emotion'] in ['disgust', 'anger', 'surprise']:
            sel = True
            for listener in turn['listener']:
                reln = listener['relation']
                if field_rev[reln] != 'family' or pos_rev[reln] != 'inferior':
                    sel = False
            if sel:
                break
    break


# ### OpenAI API

# In[25]:


import openai

openai.api_key = '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'

def get_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256
    )
    return response

def get_classification(query):
    response = openai.Classification.create(
        examples = emotion_class_examples,
        labels = list(emotion_set),
        query = query,
        search_model = "davinci",
        model = "davinci",
        return_prompt = True
    )
    return response


# In[26]:


print(emotion_set)
emotion_prompt = 'What is the emotion of the speaker? Choose one option from '
for em in emotion_set:
    emotion_prompt += em + ', '
emotion_prompt = emotion_prompt[:-2]
print(emotion_prompt)


# In[27]:


emotion_ques = []
emotion_ans = []
for conv_id in dialogue:
    conv = dialogue[conv_id]
    for i, turn in enumerate(conv):
        turn_prompt = 'Dialogue ID: ' + conv_id + '-' + str(i) + '\n'
        turn_prompt += 'Speaker: ' + turn['speaker'] + '\n'
        turn_prompt += 'Listeners: ' + str(turn['listener']) + '\n'
        turn_prompt += 'Utterance: ' + turn['utterance'] + '\n'
        emotion_ques.append(turn_prompt + emotion_prompt)
        emotion_ans.append(turn['emotion'])


# In[28]:


import random
random.seed(4056)
sel_ids = random.sample(list(range(len(emotion_ques))), 100)


# In[29]:


sel_ques = [emotion_ques[i] for i in sel_ids]
sel_ans = [emotion_ans[i] for i in sel_ids]


# In[30]:


emotion_class_examples = [[sel_ques[i].split('\n')[3].split(':')[1].strip(), sel_ans[i]] for i in range(50)]


# In[31]:


print(sel_ques[50])


# In[36]:


preds = []
golds = []
for idx in [51]:
    print(sel_ques[idx].strip())
    pred = get_response(sel_ques[idx].strip())
    print(pred['choices'][0]['text'])
    # gold = sel_ans[idx]
    # preds.append(pred)
    # golds.append(gold)

