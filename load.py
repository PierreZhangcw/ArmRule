#coding:utf8
import csv
import json
import pickle

def load_pickle(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def write_json(data,path):
    with open(path,'w') as f:
        json.dump(data,path,indent=1)

def write_beta(beta,path):
    f = open(path,'w',newline = '')
    writer = csv.writer(f)
    for z in beta:
        writer.writerow([z[0],z[1],beta[z]])
    f.close()

def write_rules(rules,path):
    f = open(path,'w',newline = '')
    writer = csv.writer(f)
    for rule in rules:
        rule = list(rule)
        row = [rule[0][0],rule[0][1]] + rule[1:]
        writer.writerow(row)
    f.close()

def write_model(model,path):
    with open(path,'wb') as f:
        pickle.dump(model,f)

def load_model(path):
    from GEMBP import GEMBP
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model
