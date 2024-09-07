import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import softmax

label_list = ['label_QCD' , 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']  

n_classes = 10 

predictions = np.load('outputs_base.npy')

labels = np.load('labels_base.npy')

y_prob = softmax(predictions, axis=1)  

scores = y_prob[:,1:10]/ (y_prob[:,0][:, np.newaxis] + y_prob[:,1:10])

scores = np.concatenate((y_prob[:,0].reshape(len(scores),1), scores), axis = 1)

rejections = []

for i in range(1, n_classes):  
    percent = 0.5
    
    mask = (labels[:, 0] == 1) | (labels[:, i] == 1)
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]
    
    binary_labels = (filtered_labels[:, i] == 1).astype(int)
    
    binary_scores = filtered_scores[:, i]
    
    fpr, tpr, thresholds = roc_curve(binary_labels, binary_scores)

    if i == 5:
        percent = 0.99
    if i == 9:
        percent = 0.995
    
    idx = np.abs(tpr - percent).argmin()
    
    if fpr[idx] != 0:
        rejection = 1 / fpr[idx]
    else:
        rejection = np.inf  
    
    rejections.append(rejection)

    
    print(f'Rejection at {percent*100}% for {label_list[i]}: {rejection}')
    
overall_roc_auc = roc_auc_score(labels, scores, average='macro', multi_class='ovo')

predicted_labels = np.argmax(y_prob, axis=1)  
true_labels = np.argmax(labels, axis=1)  
accuracy = accuracy_score(true_labels, predicted_labels)

print(f'Overall ROC AUC = {overall_roc_auc:.4f}, Accuracy = {accuracy:.4f}')

