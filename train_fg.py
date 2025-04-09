#coding:utf8
import multiprocessing
import sys

from tqdm import tqdm
#from em1 import *
from utility import *
##########
from configuration import *

import random
import logging

import numpy as np

from functools import reduce
#from tqdm import tqdm
from scipy import sparse

#from configuration import *
from load import *
from calcul import *

# using confidence as initial values of arm->arm rules
#beta_t0 = load_pickle(root+'beta_t.pickle')
#beta_f0 = load_pickle(root+'beta_f.pickle')

#####class for the factor graph#########
class graph:
    '''
    attributes:
        gamma,gamma0,gamma1,Lambda,beta0: float range (0,1) [hyper-parameter]
        X: dict{tuple:int}         [all triples and their ids]
        X_T: set(int)              [positive triple ids]
        X_F: set(int)              [negative triple ids]
        AA: dict{tuple:int}        [positive arm-arm rules:id]
        AA_: dict{tuple:int}       [negative arm-arm rules:id]
        beta: dict{int:float}      [posterior probability of ]
        beta_: dict{int:float}     [posterior probability of ]
        RA = dict{str:int}         [positive relation-arm rules:id]
        RA_ = dict{str:int}        [negative relation-arm rules:id]
        RR = dict{str:int}         [positive relation-relation rules:id]
        RR_ = dict{str:int}        [negative relation-arm rules:id]
        CR = dict{str:int}         #[positive relation-arm rules:id]
        message: sparse matrix,只记录传递值为 1 时的信息
    '''
    global train_data,triples
    global config
    #global beta_t0,beta_f0

    def __init__(self,arm_rules):
        self.gamma = config.GAMMA
        self._gamma = 1.0 - self.gamma
        self.gamma0 = config.GAMMA0
        self.gamma1 = config.GAMMA1
        self.beta0 = 0.5
        self.Lambda = config.LAMBDA
        self._Lambda = 1.0 - self.Lambda
        self.arm_rules = arm_rules
        self.parse()

    def parse(self):
        '''
        Create nodes and edges for factor graph
        :param arm_rules:
        :return:
        '''
        logging.info("preprocessing......")
        # target <arm> and <relation> of the rules in this sub-factor graph
        self.target_arm = self.arm_rules[0]
        self.target_r = self.target_arm[0]

        # arms in rule body of positive rules and negative rules
        self.body_arm = {rule[0] for rule in self.arm_rules[1]['pos']}
        self.body_arm_ = {rule[0] for rule in self.arm_rules[1]['neg']}
        self.body_arms = self.body_arm|self.body_arm_

        # relations in rule body of positive rules and negative rules
        self.body_r = set([a[0] for a in self.body_arm])
        if self.target_r in self.body_r:
            self.body_r.remove(self.target_r)
        self.body_r_ = set([a[0] for a in self.body_arm_])
        if self.target_r in self.body_r_:
            self.body_r_.remove(self.target_r)
        self.body_rs = self.body_r|self.body_r_

        # <head entities> related to rule bodies
        body_arm_data = train_data[train_data['arm'].isin(self.body_arm)]
        body_arm_data_ = train_data[train_data['arm'].isin(self.body_arm_)]
        self.body_arm_head = set(body_arm_data['head'].values)
        self.body_arm_head_ = set(body_arm_data_['head'].values)
        self.body_arm_heads = self.body_arm_head|self.body_arm_head_
        self.body_arm2head = {}
        for v in body_arm_data[['arm','head']].values:
            self.body_arm2head.setdefault(v[0],set()).add(v[1])
        self.body_arm2head_ = {}
        for v in body_arm_data_[['arm','head']].values:
            self.body_arm2head_.setdefault(v[0],set()).add(v[1])

        body_r_data = train_data[train_data['relation'].isin(self.body_r)]
        body_r_data_ = train_data[train_data['relation'].isin(self.body_r_)]
        self.body_r_head = set(body_r_data['head'].values)
        self.body_r_head_ = set(body_r_data_['head'].values)
        self.body_r_heads = self.body_r_head|self.body_r_head_
        self.body_r2head = {}
        for v in body_r_data[['relation', 'head']].values:
            self.body_r2head.setdefault(v[0], set()).add(v[1])
        self.body_r2head_ = {}
        for v in body_r_data_[['relation', 'head']].values:
            self.body_r2head_.setdefault(v[0], set()).add(v[1])

        # <head entities> related to target arm and relation
        self.target_arm_heads = set(train_data[train_data['arm']==self.target_arm]['head'].values)
        self.target_r_heads = set(train_data[train_data['relation']==self.target_r]['head'].values)

        self.rule_types = {'aa','aa_',
                           'ar','ar_',
                           'ra','ra_',
                           'rr','rr_'}
        self.rule_conflict = {
            'aa':{'ra','ar','rr','ra_','rr_'},
            'aa_':{'ra','ra_','ar_','rr_'},
            'ra':{'aa','ar','rr','aa_','ar_','ra_','rr_'},
            'ra_':{'aa','ra','aa_','ar_','rr_'},
            'ar':{'aa','ra','rr','rr_'},
            'ar_':{'ra','rr','aa_','ra_','rr_'},
            'rr':{'aa','ra','ar','ar_','rr_'},
            'rr_':{'aa','ar','ra','rr','aa_','ar_','ra_'}
        }
        # create nodes for factor graph
        logging.info("create nodes for factor graph......")
        self.create_nodes()

        # initialize posterior probability of rules
        self.init_proba_for_rules()

        # create edges for factor graph (form of messages)
        logging.info("create edges for graph......")
        self.create_edges()

    def create_nodes(self):
        '''
        create positive and negative triple nodes:(h,arm) & ~(h,arm)
        :return:
        '''
        target_heads = self.body_r_heads|self.body_arm_heads
        target_triple_heads = self.target_arm_heads&target_heads
        target_triple_heads_ = target_heads - self.target_arm_heads
        self.target_triple = set([(h,self.target_arm) for h in target_triple_heads])
        self.target_triple_ = set([(h,self.target_arm) for h in target_triple_heads_])

        # if len(self.X_F) >20*len(self.X_T):
        #     self.X_F = set(random.sample(self.X_F,len(self.X_T)*10))
        logging.info("create node id......")
        self.trans2id()

    def trans2id(self):
        '''
        make identifier for +,- triples and rules
        also for functional rel node
        '''
        def id_for_rule(obj_set):
            return dict([rule,i] for i,rule in enumerate(list(obj_set)))

        self.nodes = {}
        # (i) AA+: arm->arm
        self.nodes['aa'] = id_for_rule(self.arm_rules[1]['pos'])
        # (ii) AA-: arm->not arm
        self.nodes['aa_'] = id_for_rule(self.arm_rules[1]['neg'])
        # (iii) RA+: relation->arm
        self.nodes['ra'] = id_for_rule(self.body_r)
        # (iv) RA-: relation->not arm
        self.nodes['ra_'] = id_for_rule(self.body_r_)
        # (v) AR+: arm->relation
        self.nodes['ar'] = id_for_rule(self.body_arm)
        # (vi) AR-: arm->not relation
        self.nodes['ar_'] = id_for_rule(self.body_arm_)
        # (vii) RR+: relation->relation
        self.nodes['rr'] = id_for_rule(self.body_r)
        # (viii) RR-: relation->not relation
        self.nodes['rr_'] = id_for_rule(self.body_r_)

        # (ix) positive and negative functional triple node <fw>
        i = 0
        self.target_triples = {}
        for triple in self.target_triple:
            self.target_triples[triple] = i
            i += 1
        self.target_id_triple = set(range(i))
        self.num_xt = i
        for triple in self.target_triple_:
            self.target_triples[triple] = i
            i += 1
        self.target_id_triple_ = set(range(len(self.target_id_triple),i))
        self.num_xf = len(self.target_triples) - self.num_xt
        # (x) functional nodes for rule-compatibility function
        self.rule_compat = id_for_rule(self.body_rs)

        logging.info("positive triples:"+str(len(self.target_id_triple)))
        logging.info("negative triples:"+str(len(self.target_id_triple_)))
        for rule_type in self.rule_types:
            logging.info(rule_type+":"+str(len(self.nodes[rule_type])))

    def init_proba_for_rules(self):

        def proba_rules(rules):
            return dict([rules[rule], random.uniform(0.0001,1)] for rule in rules)

        self.beta = {}
        for rule_type in self.rule_types:
            self.beta[rule_type] = proba_rules(self.nodes[rule_type])

    def create_edges(self):
        '''
        construct relationships between nodes
        - between rules and triples
        - between rules
        :return:
        '''
        #logging.info('Create edges......')
        keys = {'x', 'compat',
                'aa', 'aa_',
                'ra', 'ra_',
                'ar', 'ar_',
                'rr', 'rr_'}
        self.messages = dict([k,{}] for k in keys)
        # create message-edges between rules and triples
        # (i) {<AA+/-> & <x>} and {<AR+/-> & <x>}

        self.messages['x']['aa'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['aa'])))
        self.messages['aa']['x'] = sparse.lil_matrix((len(self.nodes['aa']), len(self.target_triples)))
        self.messages['x']['ar'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['ar'])))
        self.messages['ar']['x'] = sparse.lil_matrix((len(self.nodes['ar']), len(self.target_triples)))
        for arm in self.body_arm:
            col = self.nodes['aa'][(arm,self.target_arm)]
            col_r = self.nodes['ar'][arm]
            for h in self.body_arm2head[arm]:
                row = self.target_triples[(h,self.target_arm)]
                self.messages['x']['aa'][row,col] = random.uniform(0.0001,1)
                self.messages['aa']['x'][col,row] = random.uniform(0.0001,1)
                self.messages['x']['ar'][row,col_r] = random.uniform(0.0001,1)
                self.messages['ar']['x'][col_r,row] = random.uniform(0.0001,1)

        self.messages['x']['aa_'] = sparse.lil_matrix((len(self.target_triples),len(self.nodes['aa_'])))
        self.messages['aa_']['x'] = sparse.lil_matrix((len(self.nodes['aa_']),len(self.target_triples)))
        self.messages['x']['ar_'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['ar_'])))
        self.messages['ar_']['x'] = sparse.lil_matrix((len(self.nodes['ar_']), len(self.target_triples)))
        for arm in self.body_arm_:
            col = self.nodes['aa_'][(arm, self.target_arm)]
            col_r = self.nodes['ar_'][arm]
            for h in self.body_arm2head_[arm]:
                row = self.target_triples[(h, self.target_arm)]
                self.messages['x']['aa_'][row, col] = random.uniform(0.0001, 1)
                self.messages['aa_']['x'][col, row] = random.uniform(0.0001, 1)
                self.messages['x']['ar_'][row,col_r] = random.uniform(0.0001,1)
                self.messages['ar_']['x'][col_r,row] = random.uniform(0.0001,1)


        # (ii) {<RA+/-> & <x>} and {<RR+/-> & <x>}
        self.messages['x']['ra'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['ra'])))
        self.messages['ra']['x'] = sparse.lil_matrix((len(self.nodes['ra']), len(self.target_triples)))
        self.messages['x']['rr'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['rr'])))
        self.messages['rr']['x'] = sparse.lil_matrix((len(self.nodes['rr']), len(self.target_triples)))
        for r in self.body_r:
            col = self.nodes['ra'][r]
            col_r = self.nodes['rr'][r]
            for h in self.body_r2head[r]:
                row = self.target_triples[(h,self.target_arm)]
                self.messages['x']['ra'][row,col] = random.uniform(0.0001,1)
                self.messages['ra']['x'][col,row] = random.uniform(0.0001,1)
                self.messages['x']['rr'][row,col_r] = random.uniform(0.0001, 1)
                self.messages['rr']['x'][col_r,row] = random.uniform(0.0001, 1)

        self.messages['x']['ra_'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['ra_'])))
        self.messages['ra_']['x'] = sparse.lil_matrix((len(self.nodes['ra_']), len(self.target_triples)))
        self.messages['x']['rr_'] = sparse.lil_matrix((len(self.target_triples), len(self.nodes['rr_'])))
        self.messages['rr_']['x'] = sparse.lil_matrix((len(self.nodes['rr_']), len(self.target_triples)))
        for r in self.body_r_:
            col = self.nodes['ra_'][r]
            col_r = self.nodes['rr_'][r]
            for h in self.body_r2head_[r]:
                row = self.target_triples[(h, self.target_arm)]
                self.messages['x']['ra_'][row, col] = random.uniform(0.0001, 1)
                self.messages['ra_']['x'][col, row] = random.uniform(0.0001, 1)
                self.messages['x']['rr_'][row,col_r] = random.uniform(0.0001, 1)
                self.messages['rr_']['x'][col_r,row] = random.uniform(0.0001, 1)

        # create message-edges between rules and triples
        # (i) <rule-compat> & <AA+/->,<AR+/->
        self.messages['compat']['aa'] = sparse.lil_matrix((len(self.body_rs),len(self.nodes['aa'])))
        self.messages['aa']['compat'] = sparse.lil_matrix((len(self.nodes['aa']),len(self.body_rs)))
        self.messages['compat']['ar'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['ar'])))
        self.messages['ar']['compat'] = sparse.lil_matrix((len(self.nodes['ar']), len(self.body_rs)))
        for arm in self.body_arm:
            r = arm[0]
            if r==self.target_r:
                continue
            row = self.rule_compat[r]
            col_a = self.nodes['aa'][(arm,self.target_arm)]
            col_r = self.nodes['ar'][arm]
            self.messages['compat']['aa'][row,col_a] = random.uniform(0.0001, 1)
            self.messages['aa']['compat'][col_a,row] = random.uniform(0.0001, 1)
            self.messages['compat']['ar'][row,col_r] = random.uniform(0.0001, 1)
            self.messages['ar']['compat'][col_r,row] = random.uniform(0.0001, 1)

        self.messages['compat']['aa_'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['aa_'])))
        self.messages['aa_']['compat'] = sparse.lil_matrix((len(self.nodes['aa_']), len(self.body_rs)))
        self.messages['compat']['ar_'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['ar_'])))
        self.messages['ar_']['compat'] = sparse.lil_matrix((len(self.nodes['ar_']), len(self.body_rs)))
        for arm in self.body_arm_:
            r = arm[0]
            if r==self.target_r:
                continue
            row = self.rule_compat[r]
            col_a = self.nodes['aa_'][(arm,self.target_arm)]
            col_r = self.nodes['ar_'][arm]
            self.messages['compat']['aa_'][row,col_a] = random.uniform(0.0001, 1)
            self.messages['aa_']['compat'][col_a,row] = random.uniform(0.0001, 1)
            self.messages['compat']['ar_'][row,col_r] = random.uniform(0.0001, 1)
            self.messages['ar_']['compat'][col_r,row] = random.uniform(0.0001, 1)

        # (i) <rule-compat> & <RA+/->,<RR+/->
        self.messages['compat']['ra'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['ra'])))
        self.messages['ra']['compat'] = sparse.lil_matrix((len(self.nodes['ra']), len(self.body_rs)))
        self.messages['compat']['rr'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['rr'])))
        self.messages['rr']['compat'] = sparse.lil_matrix((len(self.nodes['rr']), len(self.body_rs)))
        for r in self.body_r:
            row = self.rule_compat[r]
            col_a = self.nodes['ra'][r]
            col_r = self.nodes['rr'][r]
            self.messages['compat']['ra'][row, col_a] = random.uniform(0.0001, 1)
            self.messages['ra']['compat'][col_a, row] = random.uniform(0.0001, 1)
            self.messages['compat']['rr'][row, col_r] = random.uniform(0.0001, 1)
            self.messages['rr']['compat'][col_r, row] = random.uniform(0.0001, 1)

        self.messages['compat']['ra_'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['ra_'])))
        self.messages['ra_']['compat'] = sparse.lil_matrix((len(self.nodes['ra_']), len(self.body_rs)))
        self.messages['compat']['rr_'] = sparse.lil_matrix((len(self.body_rs), len(self.nodes['rr_'])))
        self.messages['rr_']['compat'] = sparse.lil_matrix((len(self.nodes['rr_']), len(self.body_rs)))
        for r in self.body_r_:
            row = self.rule_compat[r]
            col_a = self.nodes['ra_'][r]
            col_r = self.nodes['rr_'][r]
            self.messages['compat']['ra_'][row, col_a] = random.uniform(0.0001, 1)
            self.messages['ra_']['compat'][col_a, row] = random.uniform(0.0001, 1)
            self.messages['compat']['rr_'][row, col_r] = random.uniform(0.0001, 1)
            self.messages['rr_']['compat'][col_r, row] = random.uniform(0.0001, 1)


    def stepE(self,iteration = 7):
        ite = 1
        while ite <= iteration:
            logging.info('BP iteration:%d'%(ite))
            # (i) <rule-compat>-><rule>
            logging.info('(i) rule-compat->rule')
            self.update_compat_rules()
            # (ii) rules->x
            logging.info('(ii) rules->x')
            self.update_rules_x()
            # (iii) X->Z
            logging.info('(iii) x->rules')
            self.update_x_rules()
            # (iv) Z->CR
            logging.info('(iv) rules->rule-comapt')
            self.update_rules_compat()
            ite += 1
        # logging.info('Update probability.')
        self.update_posterior_proba()


    def update_rules_x(self):
        for rule_type in self.rule_types:
            for rule_id in self.nodes[rule_type].values():
                infos = self.mes_in_rule_node(rule_type,rule_id)
                for x in self.messages[rule_type]['x'].rows[rule_id]:
                    m1 = self.messages['x'][rule_type][x,rule_id]
                    self.messages[rule_type]['x'][rule_id,x] = sum_res(infos[0]/(1.0-m1),infos[1]/m1)


    def update_rules_compat(self):
        for rule_type in self.rule_types:
            for rule_id in self.nodes[rule_type].values():
                infos = self.mes_in_rule_node(rule_type,rule_id)
                for r in self.messages[rule_type]['compat'].rows[rule_id]:
                    m1 = self.messages['compat'][rule_type][r,rule_id]
                    self.messages[rule_type]['compat'][rule_id,r] = sum_res(infos[0]/(1.0-m1),infos[1]/m1)


    def update_x_rules(self):
        _gamma0 = 1 - self.gamma0
        _gamma1 = 1 - self.gamma1
        self.c10 = self.gamma0 / (self.gamma1 - self.gamma0)
        self.c11 = self.gamma1 / (self.gamma1 - self.gamma0)
        self.c00 = _gamma0 / (self.gamma1 - self.gamma0)
        self.c01 = _gamma1 / (self.gamma1 - self.gamma0)
        for x in self.target_triples.values():
            self.x_rules(x)

    def x_rules(self,x):
        x_t_in = []
        x_f_in = []
        for rule_type in self.rule_types:
            if '_' in rule_type:
                x_f_in += [self.messages[rule_type]['x'][rule_id,x]\
                           for rule_id in self.messages['x'][rule_type].rows[x]]
            else:
                x_t_in += [self.messages[rule_type]['x'][rule_id,x]\
                           for rule_id in self.messages['x'][rule_type].rows[x]]

        if len(x_f_in) == 0:
            m0 = 1
        else:
            m0 = reduce(lambda x1, x2: x1 * x2, 1 - np.array(x_f_in))
        if len(x_t_in) == 0:
            m1 = 1
        else:
            m1 = reduce(lambda x1, x2: x1 * x2, 1 - np.array(x_t_in))

        for rule_type in self.rule_types:
            for rule_id in self.nodes[rule_type].values():
                self.sub_x_rules(x,rule_type,rule_id,m0,m1)


    def sub_x_rules(self,x,rule_type,rule_id,m0,m1):
        if x<self.num_xt:
            if '_' in rule_type:
                m0 = m0/(1-self.messages[rule_type]['x'][rule_id,x])
                mes0 = m0 * m1 * (self.c10 + self.gamma) + \
                       m0 * (1 - m1) * (self.Lambda * self.c11 + self._Lambda * (self.c10 + self.gamma)) + \
                       (1 - m0) * m1 * (self.Lambda * self.c10 + self._Lambda * (self.c10 + self.gamma))
                mes1 = m1 * (self.Lambda * self.c10 + self._Lambda * (self.c10 + self.gamma))
                self.messages['x'][rule_type][x,rule_id] = sum_res(mes0, mes1)
            else:
                m1 = m1/(1 - self.messages[rule_type]['x'][rule_id,x])
                mes0 = m0*m1*(self.c10+self.gamma)+\
                       m0*(1-m1)*(self.Lambda*self.c11+self._Lambda*(self.c10+self.gamma))+\
                       (1-m0)*m1*(self.Lambda*self.c10+self._Lambda*(self.c10+self.gamma))
                mes1 = m0*(self.Lambda*self.c11+self._Lambda*(self.c10+self.gamma))
                self.messages['x'][rule_type][x, rule_id] = sum_res(mes0, mes1)

        else:
            if '_' in rule_type:
                m0 = m0/(1-self.messages[rule_type]['x'][rule_id,x])
                mes0 = m0 * m1 * (self.c00 - self.gamma) + \
                       m0 * (1 - m1) * (self.Lambda * self.c01 + self._Lambda * (self.c00 - self.gamma)) + \
                       (1 - m0) * m1 * (self.Lambda * self.c00 + self._Lambda * (self.c00 - self.gamma))
                mes1 = m1*(self.Lambda * self.c00 + self._Lambda * (self.c00 - self.gamma))
                self.messages['x'][rule_type][x,rule_id] = sum_res(mes0, mes1)
            else:
                m1 = m1/(1 - self.messages[rule_type]['x'][rule_id,x])
                mes0 = m0 * m1 * (self.c00 - self.gamma) + \
                       m0 * (1 - m1) * (self.Lambda * self.c01 + self._Lambda * (self.c00 - self.gamma)) + \
                       (1 - m0) * m1 * (self.Lambda * self.c00 + self._Lambda * (self.c00 - self.gamma))
                mes1 = m0*(self.Lambda * self.c01 + self._Lambda * (self.c00 - self.gamma))
                self.messages['x'][rule_type][x, rule_id] = sum_res(mes0, mes1)

    def update_compat_rules(self):

        def calcul_mes0(remain_types,m0):
            if remain_types:
                target_type = remain_types.pop()
                mes1 = 1.0
                for conflict_type in self.rule_conflict[target_type]:
                    if conflict_type in remain_types:
                        mes1 *= m0[conflict_type]
                return mes1*(1.0 - m0[target_type]) + \
                       calcul_mes0(remain_types,m0)*m0[target_type]
            else:
                return 1.0


        for r in self.rule_compat.values():
            # rule vector = 0 or not
            m0 = {}
            for rule_type in self.rule_types:
                m0[rule_type] = 1
                for rule_id in self.messages['compat'][rule_type].rows[r]:
                    m0[rule_type] *= (1.0 - self.messages[rule_type]['compat'][rule_id,r])

            for rule_type in self.rule_types:
                mes1 = 1.0
                for conflict_type in self.rule_conflict[rule_type]:
                    mes1 *= m0[conflict_type]
                remain_types = list(self.rule_types)
                remain_types.remove(rule_type)
                infos = calcul_mes0(remain_types,m0)
                for rule_id in self.messages['compat'][rule_type].rows[r]:
                    local_m0 = m0[rule_type]/(1.0 - self.messages[rule_type]['compat'][rule_id,r])
                    mes0 = local_m0*infos + (1.0 - local_m0)*mes1
                    self.messages['compat'][rule_type][r,rule_id] = sum_res(mes0,mes1)


    def update_posterior_proba(self):
        for rule_type in self.rule_types:
            for rule_id in self.nodes[rule_type].values():
                infos = self.mes_in_rule_node(rule_type,rule_id)
                self.beta[rule_type][rule_id] = sum_res(infos[0],infos[1])


    def mes_in_rule_node(self,rule_type,rule_id):
        z0 = 1.0 - self.beta0
        z1 = self.beta0

        for x in self.messages[rule_type]['x'].rows[rule_id]:
            z1 *= self.messages['x'][rule_type][x,rule_id]
            z0 *= (1.0 - self.messages['x'][rule_type][x,rule_id])

        for r in self.messages[rule_type]['compat'].rows[rule_id]:
            z1 *= self.messages['compat'][rule_type][r,rule_id]
            z0 *= (1.0 - self.messages['compat'][rule_type][r,rule_id])
        return {0:z0,1:z1}

'''
    def getrules(self,threshold):
        dic1 = dict([k,v] for v,k in self.Z_T.items())
        dic2 = dict([k,v] for v,k in self.Z_F.items())
        pos_rules = []
        for z in self.beta_T:
            if self.beta_T[z]>threshold:
                pos_rules += [(dic1[z],self.beta_T[z])]

        neg_rules = []
        for z in self.beta_F:
            if self.beta_F[z]>threshold:
                neg_rules += [(dic2[z],self.beta_F[z])]
        return [pos_rules,neg_rules]

    def get_beta(self):
        dic1 = dict([k, v] for v, k in self.Z_T.items())
        dic2 = dict([k, v] for v, k in self.Z_F.items())
        zT = dict([dic1[z],self.beta_T[z]] for z in self.beta_T)
        zF = dict([dic2[z],self.beta_F[z]] for z in self.beta_F)

        dic3 = dict([k, v] for v, k in self.R_T.items())
        dic4 = dict([k, v] for v, k in self.R_F.items())
        rT = dict([dic3[r],self.beta_rt[r]] for r in self.beta_rt)
        rF = dict([dic4[r],self.beta_rf[r]] for r in self.beta_rf)
        return [zT,zF,rT,rF]
    def save_model(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)

    def score_y(self,x):
        p_pos = [self.message_ztx[z,x] for z in self.message_xzt.rows[x]]
        p_pos += [self.message_rtx[r,x] for r in self.message_xrt.rows[x]]
        p_neg = [self.message_zfx[z,x] for z in self.message_xzf.rows[x]]
        p_neg += [self.message_rfx[r,x] for r in self.message_xrf.rows[x]]
        m1 = reduce(lambda x,y:x*y,1-np.array(p_pos),1)
        m0 = reduce(lambda x,y:x*y,1-np.array(p_neg),1)
        #计算w的分�?
        w0 = m1 * (1 - m0) * self.Lambda
        w1 = m0 * (1 - m1) * self.Lambda
        w2 = m0 * m1 + m0 * (1 - m1) * (1-self.Lambda) + m1 * (1 - m0) * (1-self.Lambda)
        # 计算y的分�?
        y0 = w0 + w2 *(1-self.gamma)
        y1 = w1 + w2 * self.gamma
        # 得到y的概�?
        s = y0 + y1
        py = y1/s
        return py

    def score_x(self,x):
        py = self.score_y(x)
        px = self.gamma0*(1.0-py)+self.gamma1*py
        return px

    def error_detection(self):
        triple_score = {}
        id2triple = dict([k, v] for v, k in self.X.items())
        for x in self.X_T:
            triple_score[id2triple[x]] = self.score_y(x)
        return [triple_score,len(self.X_T),len(self.X_F)]
'''

def EMBP(arm_rules):
    G = graph(arm_rules)
    for i in range(2):
        #G.stepE()
        #G.stepM()
        #logging.info('StepE iteration:%d'%(i+1))
        G.stepE(iteration=3)
        #G.stepM()
    # if config.error_detection==True:
    #     triple_score = G.error_detection()
    #     return G.get_beta()+[arm_rules[0]]+triple_score
    # else:
    #     return G.get_beta()+[arm_rules[0]]#[{'lambda':G.Lambda,'gamma':G.gamma,'gamma0':G.gamma0,'gamma1':G.gamma1},arm_rules[0]] #[G.beta0,G.Lambda,arm_rules[0]]




if __name__=='__main__':
    config = Config()#task = 'linkprediction')
    #if len(sys.argv) == 1:
        #print("need argv")
        #exit()
    #config.DATA_PATH = '../ficdata2/'+sys.argv[1]+'/'
    #config.TASK_PATH = '../ficdata2/'+sys.argv[1]+'/'
    config.DATA_PATH = '../nell995/'
    config.TASK_PATH = '../nell995/linkprediction/'
    ratio = 0#float(sys.argv[1])
    #####train data###########
    train_data = load_pickle(config.DATA_PATH+'train.pickle')
    # train_data = load_pickle(config.TASK_PATH+'trainWithNoise'+str(ratio)+'.pickle')
    triples = set([tuple(triple) for triple in train_data[['head','arm']].values])

    ###Part 1: Training the model and find the error triples.###
    candidate_rules = load_pickle(config.DATA_PATH + 'candidate_rules.pickle')
    headRules,tailRules = armToRule(candidate_rules)
    logging.info('There are %d arms which can be affected by candidate rules.',len(tailRules))
    #训练的时候使用多进程
    logging.info('Start training......')
    logging.info('Configuration:'+config.print_config())
    # ****test****** #
    example = list(tailRules.items())[0]
    EMBP(example)
    # ****test****** #
    # pool = multiprocessing.Pool(processes=config.cores)
    # results = pool.map(EMBP,tqdm(list(tailRules.items ())))
    # pool.close()
    # pool.join()
    # logging.info('Integrate all the results.')
    # beta_t = {}
    # beta_f = {}
    # r_t = {}
    # r_f = {}
    #triple_score = {}
    #theta = {}
    #cc0,cc1 = 0,0
    # for res in tqdm(results):
    #     for rule in res[0]:
    #         beta_t[rule] = res[0][rule]
    #     for rule in res[1]:
    #         beta_f[rule] = res[1][rule]
    #     for r in res[2]:
    #         if r not in r_t:
    #             r_t[r] = {}
    #         r_t[r][res[4]] = res[2][r]
    #     for r in res[3]:
    #         if r not in r_f:
    #             r_f[r] = {}
    #         r_f[r][res[4]] = res[3][r]
       # theta[res[-1]] = res[-2]
        #triple_score.update(res[5])
        #cc1+=res[6]
        #cc0+=res[7]
    #print(cc1,cc0)
    logging.info('Training over......')
    ##write_pickle(beta_t,config.TASK_PATH + 'p_beta_t.pickle')
    ##write_pickle(beta_f,config.TASK_PATH + 'n_beta_f.pickle')
    ##write_pickle(r_t,config.TASK_PATH + 'r_t.pickle')
    ##write_pickle(r_f,config.TASK_PATH + 'r_f.pickle')
    #write_pickle(theta,config.TASK_PATH+'theta.pickle')
    #len(triple_score)
    #write_pickle(triple_score,config.DATA_PATH+'errordetection/tripleScore'+str(ratio)+'.pickle')
    #write_pickle(triple_score,config.DATA_PATH+'errordetection/ceshitripleScore'+str(ratio)+'.pickle')

