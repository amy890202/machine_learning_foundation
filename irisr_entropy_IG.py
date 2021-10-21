# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:20:43 2021

@author: user
"""
import math
import datasets

def entropy(p1,n1):#算亂度
    if(p1==0 and n1==0):
        return 1
    elif(p1==0 or n1==0):
        return 0
    else:
        pp = p1/(p1+n1)
        np = n1/(p1+n1)
        return -pp*math.log2(pp)-np*math.log2(np)
    


def IG(p1,n1,p2,n2):#計算information gain
    num = p1 + n1 + p2 +n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)

print(IG(3,3,2,2))#3+3- 2+2-
print(IG(18,33,11,2))


print(entropy(100,0))
print(entropy(10,10))
print(entropy(29,35))#2計算29+ 35- 的資料亂度

data = datasets.load_iris()
#list的第0個點是根節點 一開始所有資料都在0
def ID3DT(feature, target):
    node = dict()
    node['data'] = range(len(target))
    tree = []
    tree.append(node)
    t = 0
    while(t<len(tree)):
        idx = tree[t]['data']
        if(sum(target[idx]==0)):
            tree[t]['leaf']=1#如果IG是0時，或全部分類都一樣就是leaf(已知分類的節點)
            tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            tree[t]['leaf']=1 
            tree[t]['decision'] = 1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):#i跑每一個特徵
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2#刀
                    G1 = []
                    G2 = []
                    for k in idx:#用刀把屬於這個code裡的全部分成兩半
                        if(feature[k][i]<thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    p1 = sum(target[G1]==1)#為1的個數
                    n1 = sum(target[G1]==0)
                    p2 = sum(target[G2]==1)
                    n2 = sum(target[G2]==0)
                    thisIG = IG(p1,n1,p2,n2)
                    if thisIG>bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres#找最好的切法
                        bestf = i
            if(bestIG>0):#如果最好的那刀大於0就要切
                tree[t]['leaf']=0
                tree[t]['selectf']= bestf
                tree[t]['threshold']= bestthres
                tree[t]['child']= [len(tree),len(tree)+1]
                #tree新增兩個node, node放進資料
                node = dict()
                node['data'] = bestG1
                tree.append(node)
                node = dict()
                node['data'] = bestG2
                tree.append(node)
            else:#如果best的那刀還是0再切也沒用
                tree[t]['leaf']=1
                if sum(target[idx]==1) > sum(target[idx]==0):#1的個數比0多
                    tree[t]['decision'] = 1
                else:
                    tree[t]['decision'] = 0
        t += 1
