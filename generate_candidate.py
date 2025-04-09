#coding:utf8
#!python3
import logging
import multiprocessing
import random
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import combinations
from collections import Counter

from configuration import *
from load import *
from utility import *


def preprocess(head):
    global data
    transaction = []
    pairCount = Counter()
    for arm in data[data['head'] == head][['relation','tail']].values:
        transaction += [tuple(arm)]
    if len(transaction)>1:
        pairs = list(combinations(transaction,2))
        for pair in pairs:
            pairCount[frozenset(pair)] = pairCount.get(frozenset(pair),0)+1
    return pairCount

def fre_r_pairs(pair):
    #step1: find relevant relation pair
    global pairCount,r_arms
    r1 = pair[0]
    r2 = pair[1]
    N = 0 # N(r1,r2)
    dic1 = dict([a,0] for a in r_arms[r1]) # r1 构成的arm的出现次数
    dic2 = dict([a,0] for a in r_arms[r2]) # r2 构成的arm的出现次数
    for a1 in r_arms[r1]:
        for a2 in r_arms[r2]:
            count = pairCount[frozenset([a1,a2])]
            N += count
            dic1[a1] += count
            dic2[a2] += count
    if N:
        score = N*np.log2(N)
        for a1 in r_arms[r1]:
            if dic1[a1]:
                score -= dic1[a1]*np.log2(dic1[a1])
            for a2 in r_arms[r2]:
                count = pairCount[frozenset([a1, a2])]
                if count:
                    score += count*np.log2(count/dic2[a2])
        score = score/N
        ##### 归一化处理
        H1,H2 = 0,0
        for a1 in r_arms[r1]:
            if dic1[a1]:
                p = dic1[a1]/N
                H1 -=  p*np.log2(p)
        for a2 in r_arms[r2]:
            if dic2[a2]:
                p = dic2[a2]/N
                H2-=  p*np.log2(p)
        H = min(H1,H2)
        if H and score:
            score = score/H
            if score>=0.2:
                return pair
    return 0

def pairs_count():
    global pairCount,r_arms,r_pairs
    arms = set()
    for pair in r_pairs:
        arms = arms | r_arms[pair[0]]
        arms = arms | r_arms[pair[1]]
    N = 0
    dic = dict([arm, 0] for arm in arms)
    for pair in r_pairs:
        r1 = pair[0]
        r2 = pair[1]
        for a1 in r_arms[r1]:
            for a2 in r_arms[r2]:
                count = pairCount[frozenset([a1, a2])]
                N += count
                dic[a1] += count
                dic[a2] += count
    print('Total pairs:', N)
    return N,dic

def fre_arm_pairs(pair):
    global r_arms,pairCount,dic,N
    r1 = pair[0]
    r2 = pair[1]
    #用阈值的方法筛选
    # pos = set()
    # neg = set()
    # for a1 in r_arms[r1]:
    #     if dic[a1]:
    #         for a2 in r_arms[r2]:
    #             if dic[a2]:
    #                 count = pairCount[frozenset([a1, a2])]
    #                 if count:
    #                     score = np.log2(count*N/(dic[a1]*dic[a2]))
    #                     if score>3:
    #                         pos.add((a1,a2))
    #                     if score<-1:
    #                         neg.add((a1,a2))
    # 用排序的方法筛选
    pos = {}
    neg = {}
    #neg['-inf'] = set()
    for a1 in r_arms[r1]:
        if dic[a1]:
            for a2 in r_arms[r2]:
                if dic[a2]:
                    count = pairCount[frozenset([a1, a2])]
                    ##########2019.02.22发现的bug##########
                    ######从未同时出现过的没考虑进去
                    ######################################
                    if count:
                        score = np.log2(count*N/(dic[a1]*dic[a2]))
                        if score>0:
                            pos[(a1,a2)] = score
                        if score<0:
                            neg[(a1,a2)] = score
                    else:
                        #neg['-inf'].add((a1,a2))
                        neg[(a1,a2)] = float('-inf')
    return [pos,neg] #arm_pairs

if __name__ == '__main__':

    # argvs = sys.argv
    # if len(argvs)>1:
    #     data_path = sys.argv[1]
    #     config = Config(data=data_path)
    # else:
    #     print('No data file given!')
    #     sys.exit()
    config = Config(data='nell995')
    ############################################################################
    #先统计所有的arm pair 及其出现的次数：pairCount:{frozenset({a1,a2}):count,...}
    ############################################################################
    input_data  = pd.read_csv(config.DATA_PATH+'triples.csv')
    heads = list(set(input_data['head'].values))
    ### 划分训练集和测试集 ###
    number_test = int(len(heads)*config.TEST_RATIO)
    test_heads = random.sample(list(heads),number_test)
    train_heads = list(set(heads) - set(test_heads))

    ### generate candidate rules from train dataset ###
    data = input_data[input_data['head'].isin(train_heads)]
    train_heads = set(data['head'].values)
    ### get <pairCout> ###
    pool = multiprocessing.Pool(processes=config.cores)
    logging.info("Geting pairCount......")
    results = pool.map(preprocess,tqdm(train_heads))
    pool.close()
    pool.join()
    pairCount = Counter()
    for res in tqdm(results):
        pairCount.update(res)
    logging.info('Number of all pairs:%d'%(len(pairCount)))
    ###############################################################################################
    #找出有关联的relation pairs.:r_pairs=[(r1,r2),...], r_arms = {r1:{a1,a2,...},r2:{b1,b2,...},...}
    ###############################################################################################
    count = data.groupby(['relation', 'tail']).size()
    logging.info("Total arms:%d,Total relations:%d"%(len(count),len(set(data['relation'].values))))

    fre_arms = list(count[count > config.minsup].index)
    logging.info("frequent arms:%d"%(len(fre_arms)))

    relations = list(set([arm[0] for arm in fre_arms]))

    r_arms = dict([r, set()] for r in relations)
    for arm in fre_arms:
        r_arms[arm[0]].add(arm)
    logging.info("relations in fre_arms:%d"%(len(relations)))
    args = list(combinations(relations,2))
    #************
    args+=[(r,r) for r in relations]
    #************
    pool = multiprocessing.Pool(processes=config.cores)
    r_pairs = pool.map(fre_r_pairs,args)
    pool.close()
    pool.join()
    r_pairs = set(r_pairs)
    if 0 in r_pairs:
        r_pairs.remove(0)
    logging.info("frequent relation pairs:%d"%(len(r_pairs)))
    ###############################################################################################
    # 找出有关联的arm pairs.:arm_pairs=[(a1,a2),...]
    ###############################################################################################
    N,dic = pairs_count()
    args = list(r_pairs)
    pool = multiprocessing.Pool(processes=config.cores)
    results = pool.map(fre_arm_pairs,args)
    pool.close()
    pool.join()

    pos_pairs = {}
    neg_pairs = {}
    for res in results:
        pos_pairs.update(res[0])
        neg_pairs.update(res[1])
    print('pos_arms_pairs:',len(pos_pairs))
    print('neg_arms_pairs:',len(neg_pairs))
    ###############################################################################################
    # 生成备选规则：pos_rules = {(a1,a2),...} neg_rules = {(a1,a2),...}
    ###############################################################################################
    sames = set([(arm,arm) for arm in fre_arms])
    for pair in sames:
        if pair in pos_pairs:
            del pos_pairs[pair]
        if pair in neg_pairs:
            del neg_pairs[pair]
    print('pos_arms_pairs:',len(pos_pairs))
    print('neg_arms_pairs:',len(neg_pairs))
    #
    # write_pickle(pos_pairs,config.DATA_PATH+'pos_pairs.pickle')
    # write_pickle(neg_pairs,config.DATA_PATH+'neg_pairs.pickle')
    #
    pos_rules = set()
    pos_pairs = sorted(pos_pairs.items(),key=lambda x:x[1],reverse = True)
    for item in pos_pairs:#[:5000]:
        pair = item[0]
        a1,a2 = pair
        pos_rules.add((a1,a2))
        pos_rules.add((a2,a1))

    neg_rules = set()
    #neg_pairs = sorted(neg_pairs.items(),key=lambda x:x[1],reverse = False)
    for item in neg_pairs:#[:2500]:
        pair = item
        a1 = pair[0]
        a2 = pair[1]
        neg_rules.add((a1,a2))
        neg_rules.add((a2,a1))

    print('positive rules:',len(pos_rules))
    print('negative rules:', len(neg_rules))

    wewant = set()
    for item in pos_pairs:#[:6000]:
        a1,a2 = item[0]
        r1,r2 =a1[0],a2[0]
        for a11 in r_arms[r1]:
            if (a2,a11) not in pos_rules:
                wewant.add((a2,a11))
        for a22 in r_arms[r2]:
            if (a1,a22) not in pos_rules:
                wewant.add((a1,a22))

    print('wewant:',len(wewant))
    for pair in sames:
        if pair in wewant:
            wewant.remove(pair)
    print('wewant:',len(wewant))
    print('wewant&neg_pairs:',len(wewant&neg_rules))
    neg_rules = set(random.sample(list(wewant&neg_rules),16000))
    pos_rules = set()
    #pos_pairs = sorted(pos_pairs.items(),key=lambda x:x[1],reverse = True)
    #cc = 0
    for item in pos_pairs:#[:5000]:
        if len(pos_rules)>=16000:
             break
        pair = item[0]
        if pair not in sames:
            #cc+=1
            a1,a2 = pair
            pos_rules.add((a1,a2))
            pos_rules.add((a2,a1))

    print('positive rules:',len(pos_rules))#,cc)
    print('negative rules:', len(neg_rules))
    # cc = 0
    # for pair in wewant:
    #     a1,a2 = pair
    #     score = 0.0
    #     if (a1,a2) in neg_pairs:
    #         score = neg_pairs[(a1,a2)]
    #     if (a2,a1) in neg_pairs:
    #         score = neg_pairs[(a2,a1)]
    #     if score==float('-inf'):
    #         cc+=1
    # print(cc)
    rules = [pos_rules,neg_rules]
    ####################################################################
    #将与备选规则有关的三元组从KB中挑出来，按照head entity分为训练集和测试集
    ####################################################################
    rules_arms = set()
    for rule in pos_rules:
        rules_arms.add(rule[0])
        rules_arms.add(rule[1])

    for rule in neg_rules:
        rules_arms.add(rule[0])
        rules_arms.add(rule[1])

    rs = set([a[0] for a in rules_arms])

    train_data = trans_data(data[data['relation'].isin(rs)])
    test_data = input_data[input_data['head'].isin(test_heads)]
    test_data = trans_data(test_data)

    root = config.DATA_PATH
    print('train_data:',len(train_data),' train triples.')
    # write_pickle(train_data,root+'train.pickle')

    print('test_data:',len(test_data),' test triples.')
    logging.info('Over!')
    # write_pickle(test_data,root+'test.pickle')

    # write_pickle(rules,root+'candidate_rules.pickle')
