

def confusion_matrix(true_labels, pred_labels, label_num):

    if len(true_labels) != len(pred_labels):
        raise ValueError("Length should be the same")
        
    confusion_matrix = [[0 for _ in range(label_num)]for _ in range(label_num)]
    for (ind,true_label) in enumerate(true_labels):
        true = int(true_label)
        
        confusion_matrix[true][int(pred_labels[ind])] += 1
    
    return confusion_matrix

