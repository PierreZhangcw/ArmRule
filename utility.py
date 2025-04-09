
#get the dict of {headArm:{pos_rule1,pos_rule2,...},...}
# from topK rules of beta_t, we use all rules ad default.
import pandas as pd
def trans_data(data):
    newdata = [[tri[0],tri[1],tri[2],(tri[1],tri[2])] for tri in data.values]
    return pd.DataFrame(newdata,columns = ['head','relation','tail','arm'])


def dic2dataframe(s,cols=['triple','score']):
    data = []
    for tri in s:
        data.append([tri,s[tri]])
    return pd.DataFrame(data,columns=cols)
    

def topk(beta,k):
    rules = sorted(beta.items(),key=lambda x:x[1],reverse=True)
    rules = rules[:k]
    dic = dict([r[0],r[1]] for r in rules)
    arm_rules = {}
    for rule in dic:
        if rule[0] not in arm_rules:
            arm_rules[rule[0]] = set()
        arm_rules[rule[0]].add(rule[1])
    return arm_rules


def armToRule(rules):
    headRules = {}
    tailRules = {}
    for rule in rules[0]:
        if rule[0] not in headRules:
            headRules[rule[0]] = {'pos': set(), 'neg': set()}
        headRules[rule[0]]['pos'].add(rule)

        if rule[1] not in tailRules:
            tailRules[rule[1]] = {'pos': set(), 'neg': set()}
        tailRules[rule[1]]['pos'].add(rule)

    for rule in rules[1]:
        if rule[0] not in headRules:
            headRules[rule[0]] = {'pos': set(), 'neg': set()}
        headRules[rule[0]]['neg'].add(rule)

        if rule[1] not in tailRules:
            tailRules[rule[1]] = {'pos': set(), 'neg': set()}
        tailRules[rule[1]]['neg'].add(rule)
    return headRules,tailRules


def evaluation(score,true,threshold=0.5):
    false = set([triple for triple in score if triple not in true])
    tp,fn = 0,0
    for triple in true:
        if score[triple]>=threshold:
            tp += 1
        else:
            fn += 1
    tn,fp = 0,0
    for triple in false:
        if score[triple]<threshold:
            tn += 1
        else:
            fp += 1
    precision = round((tp+tn)/(len(score)),4)
    p1,p0 = 1.0,1.0
    if tp+fp!=0:
        p1 = round(tp/(tp+fp),4)
    r1 = round(tp/len(true),4)
    if tn+fn!=0:
        p0 = round(tn/(tn+fn), 4)
    r0 = round(tn/len(false), 4)
    F1 = 0
    if p1+r1!=0:
        F1 = round(2*p1*r1/(p1+r1),4)
    f1 = 0
    if p0+r0!=0:
        f1 = round(2*p0*r0/(p0+r0),4)
    res = [p1,r1,F1,p0,r0,f1,precision,threshold]
    res = [str(item) for item in res]
    return '| '.join(res)
def get_f1(p,r):
    f1 = 0
    if p+r!=0:
        f1 = 2*p*r/(p+r)
    return f1
def get_hits(score,true,path):
    #########预测正例#################
    import csv
    from tqdm import tqdm
    score = sorted(score.items(),key=lambda x:x[1],reverse = True)
    N = len(score)
    T = len(true)
    F = N - len(true)
    print('all:',N,'true:',T,'false:',F)
    '''
    TP: 真正例，hits&true
    FP: 假正例，hits$false = hits-htis&true
    TN: 真负例，false-hits&false
    FN: 假负例，true-hits&true
    '''
    f = open(path,'w',newline='')
    writer = csv.writer(f)
    writer.writerow(['r+','p+','F1+',
                     'r-','p-','F1-',
                     'r','p','F1',
                     'FPR','TPR'])
    # writer.writerow([1.0,0.0,0.0,
    #                  F/N,1.0,2*F/(F+N),
    #                  F/N,F/N,F/N,
    #                  F/N,1.0,])
    hits = set()
    for i,tri in tqdm(enumerate(score)):
        hits.add(tri[0])
        TP = len(true&hits)
        FP = i+1-TP
        TN = F - FP
        FN = T - TP
        p1 = TP/(i+1)
        r1 = TP/T
        p0 = 0
        if (TN+FN)!=0:
            p0 = TN/(TN+FN)
        r0 = TN/F
        p = (TP+TN)/N
        r = (TP+TN)/N
        writer.writerow([r1,p1,get_f1(p1,r1),
                         r0,p0,get_f1(p0,r0),
                         r,p,get_f1(p,r),
                         FP/F,TP/T])
    f.close()