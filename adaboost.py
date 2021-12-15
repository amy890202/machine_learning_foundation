# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:38:08 2021
@author: USER
"""

import numpy as np
import math
import matplotlib.pyplot as plt
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']
# face1 = trainface[1500,:].reshape((19,19))
# plt.imshow(face1,cmap='gray')
trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

ftable = []
for y in range(19):#block符合就append進來
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*1-1<=18 and x+w*2-1<=18):
                    ftable.append([0,y,x,h,w])
                if(y+h*2-1<=18 and x+w*1-1<=18):
                    ftable.append([1,y,x,h,w])
                if(y+h*1-1<=18 and x+w*3-1<=18):
                    ftable.append([2,y,x,h,w])
                if(y+h*2-1<=18 and x+w*2-1<=18):
                    ftable.append([3,y,x,h,w])
fn = len(ftable)

#sample N*361(19*19的照片)->n*36648(個特徵)

def fe(sample,ftable,c):
    ftype,y,x,h,w = ftable[c]
    T = np.arange(361).reshape((19,19))
    if ftype==0:#A類型
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)#y:y+h,x:x+w為白色的區域 這些clolumn的值拿出來(變n*9)sum   白色-黑色
    elif ftype==1:
        output = np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)
    elif ftype==2:
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)+np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis=1)
    else:
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)-np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)  
    return output
#正樣本矩陣跟負樣本矩陣乾淨的分成兩類
def WC(pw,nw,pf,nf):#postive weight,negative weight,positive feature,negative feature
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    polarity = 1#是1時就假設右正左負
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])#pf<theta的人的postive weight sum起來+
    if error>0.5:
        error = 1-error
        polarity = 0
    min_theta,min_error,min_polarity = theta,error,polarity
    for i in range(2,10):#第2刀到第9刀
        theta = (maxf-minf)*i/10+minf
        polarity = 1
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        if error>0.5:
            error = 1-error
            polarity = 0
        if error<min_error:
            min_theta,min_error,min_polarity = theta,error,polarity
    return min_error,min_theta,min_polarity
        
trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))#不要用append動態長大比較有效率
tepf = np.zeros((tepn,fn))
tenf = np.zeros((tenn,fn))
for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    tepf[:,c] = fe(testface,ftable,c)
    tenf[:,c] = fe(testnonface,ftable,c)
pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2
#%%
SC = []
Tl =[1,5,20]
Tl_record = np.zeros((len(Tl),5)).tolist()
Tl_record[0][4] = 1
Tl_record[1][4] = 5
Tl_record[2][4] = 20

x_num = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


#T = 10
for numb in range(len(Tl)):
    T = Tl[numb]
    tpr_list = []
    fpr_list = []
    for t in range(T):
        weightsum = np.sum(pw)+np.sum(nw)
        pw = pw/weightsum
        nw = nw/weightsum#權重
        best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
        best_feature = 0
        for i in range(1,fn):
            error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i])
            if error<best_error:
                best_error,best_theta,best_polarity = error,theta,polarity
                best_feature = i #第i個feature就比較好
        beta = best_error / (1-best_error)
        alpha = math.log10(1/beta)#弱強化器的
        SC.append([best_feature,best_theta,best_polarity,alpha])
        #做權重 分對的*beta
        if best_polarity == 1:
            pw[trpf[:,best_feature]>=best_theta]*=beta
            nw[trnf[:,best_feature]<best_theta]*=beta
        else:
            pw[trpf[:,best_feature]<best_theta]*=beta
            nw[trnf[:,best_feature]>=best_theta]*=beta
    
    print('SC',SC)
    trps = np.zeros((trpn,1))
    trns = np.zeros((trnn,1))
    alpha_sum = 0
    for i in range(T):
        feature,theta,polarity,alpha = SC[i]
        if polarity == 1:
            trps[trpf[:,feature]>=theta] += alpha
            trns[trnf[:,feature]>=theta] += alpha
        else:
            trps[trpf[:,feature]<theta] += alpha
            trns[trnf[:,feature]<theta] += alpha
        alpha_sum += alpha
    trps = trps/alpha_sum
    trns = trns/alpha_sum
    print(np.sum(trps>=0.5)/trpn)#對的說是對的個數->ratio(true positive rate)
    print(np.sum(trps<0.5)/trpn)#對的說是錯的ratio
    print(np.sum(trns>=0.5)/trnn)#錯的說對
    print(np.sum(trns<0.5)/trnn)#錯的說錯
    tpr  = np.sum(trps>=0.5)/trpn
    fpr = np.sum(trns>=0.5)/trnn
    for x_n in x_num:
        tpr_list.append(np.sum(trps>=x_n)/trpn)
        fpr_list.append(np.sum(trns>=x_n)/trnn)
    train_accuracy = (np.sum(trps>=0.5)/trpn + np.sum(trns<0.5)/trnn)/2
    Tl_record[numb][0] = tpr_list
    Tl_record[numb][1] = fpr_list
    Tl_record[numb][2] = train_accuracy
    
    
    #test_accuracy
    teps = np.zeros((tepn,1))
    tens = np.zeros((tenn,1))
    alpha_sum = 0
    for i in range(T):
        feature,theta,polarity,alpha = SC[i]
        if polarity == 1:
            teps[tepf[:,feature]>=theta] += alpha
            tens[tenf[:,feature]>=theta] += alpha
        else:
            teps[tepf[:,feature]<theta] += alpha
            tens[tenf[:,feature]<theta] += alpha
        alpha_sum += alpha
    teps = teps/alpha_sum
    tens = tens/alpha_sum
    print(np.sum(teps>=0.5)/tepn)#對的說是對的個數->ratio(true positive rate)
    print(np.sum(teps<0.5)/tepn)#對的說是錯的ratio
    print(np.sum(tens>=0.5)/tenn)#錯的說對
    print(np.sum(tens<0.5)/tenn)#錯的說錯
    test_accuracy = (np.sum(teps>=0.5)/tepn + np.sum(tens<0.5)/tenn)/2
    Tl_record[numb][3] = test_accuracy     

#%%


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = model.predict_proba(X_test)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
#roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt

# =============================================================================
# for t in tpr_list:
#     tpr_list.append(x_num)
# for t in fpr_list:
#     tpr_list.append(x_num)
# =============================================================================
    
plt.title('Receiver Operating Characteristic')

i = []
for t in Tl_record: 
    tpr = t[0]
    fpr = t[1]
    print( 'hi',tpr, fpr,t[4])
    #roc_auc = metrics.auc(fpr, tpr)
    plt.plot( fpr, tpr,label = 'type: %d' % t[4])#%d' % t[4]
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
#plt.legend(loc = 'lower right')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = model.predict_proba(X_test)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
#roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt

#%%
train_list = []
test_list = []
x = []
for t in Tl_record:
    train_list.append(t[2])
    test_list.append(t[3])
    x.append(t[4])
    # train = t[2]
    #test = t[3]
    #x = t[4]
plt.plot(x, train_list, label = 'train', color = 'red')
plt.plot(x, test_list, label = 'test', color = 'blue')
plt.xlim([1, 20])
plt.ylim([0, 1])
plt.ylabel('accuracy')
plt.xlabel('T')
plt.show()
