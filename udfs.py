"""
James Pagnini
Class: CS 677
Date: JUL-7-2022
Term Project
Description of Problem:
    Contains any user-defined functions for the term project.
"""

import pandas as pd
# import numpy as np

   

def measure(confusion_matrix, method):
    '''
    
    Parameters
    ----------
    confusion_matrix : numpy.ndarray 'int64'
        Expects a confusion matrix.
    method: string
        The name of the method used to model the data.

    Returns
    -------
    df : pandas dataframe
        A table of accuracy measures based on the confusion matrix, and the
        name of the method used to model the data.

    '''
    
    # Simple reference for readability.
    c = confusion_matrix
    
    # Calculate accuracy measures.
    TN = c[0][0]
    FN = c[1][0]
    FP = c[0][1]
    TP = c[1][1]
    TPR = TP / (TP + FN)
    PPV = TP / (TP + FP)
    TNR = TN / (TN + FP)
    Acc = (TP + TN) / (TP + TN + FP + FN)

    f1_score = 2*TP / (2*TP+FP+FN)

    # Create dataframe.
    df = pd.DataFrame({'Method':method,'TP':TP,'FP':FP,'TN':TN,'FN':FN,
                       'Accuracy':Acc,'f1_score':f1_score,
                       'TPR(recall)':TPR,'PPV(precision)':PPV,
                       'TNR(specificity)':TNR},
                      index=[0])

    return df





































