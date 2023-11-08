"""
James Pagnini
Class: CS 767
Date: NOV-4-2023
Homework #1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from udfs import measure
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneOut, \
                                    cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import tree


'''
Read data and split into scaled and endoded training/testing.
'''

# Read data into a pandas dataframe.
df = pd.read_csv('Pokemon.csv')
# Preserve a copy of the original data set.
df_raw = df.copy()

# Create a dataframe to store accuracy results for each model.
res = pd.DataFrame()

# Drop number and Name columns b/c their values are unique by definition.
df = df.drop(['#','Name','Type 1','Type 2','Generation'],axis=1)

# Create scaler object and label encoder object.
scaler = StandardScaler()
le = LabelEncoder()




# Scale and assign features to X. Encode and assign labels to Y.
df['Legendary'] = le.fit_transform(df['Legendary'])
X = scaler.fit_transform(df.drop(['Legendary'],axis=1))
Y = df['Legendary']

# Split X,Y into training/testing.
# Stratify to account for high ratio of class0:class1
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=.5,
                                                 random_state=0, stratify=(Y))


'''
Logistic Regression
'''

# Create a logistic regression classifier and fit it to training data.
log_reg_classifier = LogisticRegression(class_weight='balanced')
log_reg_classifier.fit(X_train,Y_train)

# Make predictions for X test data.
pred_lr = log_reg_classifier.predict(X_test)
# Calculate accuracy.
acc_lr = np.mean(pred_lr == Y_test)

# Calculate the confusion matrix for the prediction.
c1 = confusion_matrix(Y_test,pred_lr)

# Create a data frame of accuracy measures.
mes_lr = measure(c1, 'Logistic_Regression')
# Update overall accuracy table.
res = pd.concat([mes_lr,res],ignore_index=True)
# Compute and record AUROC.
res['AUC'] = np.NaN
res['AUC'].iat[0] = roc_auc_score(Y_test, 
                           log_reg_classifier.predict_proba(X_test)[:, 1])

# '''
# k Nearest Neighbor
# '''

# # Plot the accuracy for each k to visualize the effect of the # of neighbors.
# # Do so for both distance types.
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title('kNN')
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('k_Neighbors')

# # Create a list of colors for plotting different learning rates.
# colors = ['blue','green','red']
# # Create a list of distance types.
# dist = ['Manhattan','Euclidean']
# # Create a dictionary to hold the highest accuracy for Ada and the 
# # corresponding number of estimators and learning rate.
# highest_knn = {'Accuracy' : -1, 'k' : -1, 'p' : -1}

# # Create a list of accuracies for kNN using euclidean distance and kNN using
# # manhattan distance.
# for p in [1,2]:

#     # Reset the accuracy list for each loop.    
#     acc_list_knn = []
    
#     for k in range(1,21,2):
        
#         # Create, train, and test the kNN classifier for current k value.
#         knn_classifier = KNeighborsClassifier(n_neighbors=k, p=p)
#         knn_classifier.fit(X_train,Y_train)
#         pred_k = knn_classifier.predict(X_test)
#         # Populate accuracy list for current model's predictions.
#         acc_list_knn.append(np.mean(pred_k == Y_test))

#         # Update the highest accuracy information if a higher accuracy occurs.
#         if acc_list_knn[-1] > highest_knn['Accuracy']:
#             highest_knn['Accuracy'] = acc_list_knn[-1]
#             highest_knn['k'] = k
#             highest_knn['p'] = p

#     # Plot the accuracy against k neighbors for the current p value.
#     plt.plot(range(1,21,2),acc_list_knn,color=colors[p-1],linestyle='dashed',
#           marker='o', markerfacecolor='black', markersize=8,label=dist[p-1])

# # Display a legend to make the graph more readable.
# ax.legend()

# # Run kNN for the most accurate k value, k*.
# knn_classifier = KNeighborsClassifier(n_neighbors=highest_knn['k'],
#                                       p=highest_knn['p'])
# knn_classifier.fit(X_train,Y_train)
# # Test classifier on testing data.
# pred_k = knn_classifier.predict(X_test)

# # Create a confusion matrix for k*
# c2 = confusion_matrix(Y_test, pred_k)

# # Create a table of measures for k* as a predictor.
# mes_knn = measure(c2,'kNN')
# # Update overall accuracy table.
# res = pd.concat([res,mes_knn],ignore_index=True)
# # Compute and record AUROC.
# res['AUC'].iat[1] = roc_auc_score(Y_test,
#                                    knn_classifier.predict_proba(X_test)[:, 1])

'''
Naive-Bayesian
'''

# Create and fit a Gaussian Naive-Bayesian classifier to training data.
nb_classifier = GaussianNB().fit(X_train, Y_train)
# Test classifier on testing data.
pred_nb = nb_classifier.predict(X_test)

# Create a confusion matrix for NB.
c3 = confusion_matrix(Y_test, pred_nb)

# Create a table of measures for the NB predictions.
mes_nb = measure(c3, 'Naive-Bayesian')

# Update overall accuracy table.
res = pd.concat([res,mes_nb], ignore_index=True)
# Compute and record AUROC.
res['AUC'].iat[1] = roc_auc_score(Y_test, 
                           nb_classifier.predict_proba(X_test)[:, 1])


'''
Decision Tree
'''

# Create and fit a Decision Tree classifier to training data.
dt_classifier = tree.DecisionTreeClassifier(criterion = 'entropy',
                                            max_depth=10,
                                            random_state=1)
dt_classifier.fit(X_train, Y_train)
# Test classifier on testing data
pred_dt = dt_classifier.predict(X_test)

# Create a confusion matrix for dt.
c4 = confusion_matrix(Y_test, pred_dt)

# Create a table of measures for the dt predictions.
mes_dt = measure(c4, 'Decision_Tree')

# Update overall accuracy table.
res = pd.concat([res, mes_dt], ignore_index=True)
# Compute and record AUROC.
res['AUC'].iat[2] = roc_auc_score(Y_test, 
                           dt_classifier.predict_proba(X_test)[:, 1])





'''
I/O Examples 1-3
'''

inp1 = df.drop(['Legendary'],axis=1) \
          .sort_values(by='Defense',ascending=False) \
          .iloc[:5,:] \
              .index.tolist()
inp2 = df.drop(['Legendary'],axis=1) \
          .sort_values(by='Total',ascending=False) \
          .iloc[:5,:] \
              .index.tolist()
inp3 = df.drop(['Legendary'],axis=1) \
          .sort_values(by='HP',ascending=False) \
          .iloc[:5,:] \
              .index.tolist()
          
out1 = dt_classifier.predict(X[inp1,])
out2 = dt_classifier.predict(X[inp2,])
out3 = dt_classifier.predict(X[inp3,])

print('Output 1: ',out1)
print('Output 2: ',out2)
print('Output 3: ',out3)


'''
Changed data (section 6)
'''

# shrink the magnitude of the features for each element labeled 1
features = df.drop('Legendary',axis=1).columns.tolist()
df6 = df.copy()
df6.loc[df['Legendary'] == 1, features] *= 0.1



'''
Changed data - Inconsistent (section 7)
'''

# generate inconsistent entries for each legendary status pokemon
delta = df[df['Legendary'] == 1].copy()
delta['Legendary'] = 0
df7 = pd.concat([df,delta],axis=0)


'''
Visualize the Decision Tree to examine change between data alterations
'''

# plt.figure(figsize=(12, 8))
# tree.plot_tree(dt_classifier, filled=True, feature_names=df.columns[:-1],
#                class_names=['Non-Legendary', 'Legendary']) 
# plt.savefig('decision_tree_Sec6.png')  
# plt.show() 




# '''
# Ensemble (AdaBoost)
# '''

# # Create graph comparing the accuracy of Ada classifiers of different learning
# # rates against the number of estimators to determine the most accuract
# # combination of estimator count and learning rate.
# plt.figure(figsize=(10,4))
# ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title('AdaBoost')
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('n_Estimators')

# # Create a base estimator for AdaBoost, using Support Vevtor Classification.
# svc = SVC(probability=True, kernel='linear', class_weight='balanced')

# # Assign counting integer for looping through colors list.
# color = -1
# # Create a dictionary to hold the highest accuracy for Ada and the 
# # corresponding number of estimators and learning rate.
# highest_ada = {'Accuracy' : -1, 'Learning_Rate' : -1, 'n_Estimators' : -1}

# # Create three lists of accuracies for Ada classifiers with varying numbers of 
# # estimators and varying learning rates.
# for l in np.arange(0.5,1.1,0.25):
    
#     # Reset the accuracy list for each loop.
#     acc_list_ada = []
#     # Increment counter variable.
#     color += 1
    
#     # Loop for each value of n_Estimators.
#     for n in range(1,13):
        
#         # Create the Ada classifier.
#         ada_classifier = AdaBoostClassifier(n_estimators=n, estimator=svc,
#                                             learning_rate=l, random_state=3)
#         # Fit the Ada classifier on training data.
#         ada_classifier.fit(X_train, Y_train)
#         # Test the Ada classifier on testing data.
#         pred = ada_classifier.predict(X_test)
#         # Populate the accuracy list with the accuracy of the current model.
#         acc_list_ada.append(np.mean(pred == Y_test))
        
#         # Update the highest accuracy information if a higher accuracy occurs.
#         if acc_list_ada[-1] > highest_ada['Accuracy']:
#             highest_ada['Accuracy'] = acc_list_ada[-1]
#             highest_ada['Learning_Rate'] = l
#             highest_ada['n_Estimators'] = n
        
#     # Plot the accuracy list for the current learning rate.
#     plt.plot(range(1,13,1),acc_list_ada,color=colors[color],
#              linestyle='solid', marker='o', markerfacecolor='black', 
#              markersize=8, label='learning rate = ' + str(l))

# # Display a legend to make the graph more readable.
# ax.legend()

# # Create the classifier from the variables with the highest accuracy from the
# # graph.
# ada_classifier = AdaBoostClassifier(n_estimators=highest_ada['n_Estimators'], 
#              estimator=svc, learning_rate=highest_ada['Learning_Rate'], 
#              random_state=3)

# # Fit to training data.
# ada_classifier.fit(X_train, Y_train)
# # Test predictions on testing data.
# pred_ada = ada_classifier.predict(X_test)

# # Create a confusion matrix for ada.
# c5 = confusion_matrix(Y_test, pred_ada)

# # Create a table of measures for the ada predictions.
# mes_ada = measure(c5, 'AdaBoost')

# # Update overall accuracy table.
# res = pd.concat([res, mes_ada], ignore_index=True)
# # Compute and record AUROC.
# res['AUC'].iat[4] = roc_auc_score(Y_test, 
#                            ada_classifier.predict_proba(X_test)[:, 1])


# '''
# Leave-One-Out Cross-Validation
# '''

# # Create an iterable list of models.
# models = [log_reg_classifier, knn_classifier, nb_classifier, dt_classifier,
#           ada_classifier]
# # Add a column for CV MSE to the overall results table.
# res['LOOCV_MSE'] = np.NaN

# # Define cross-validation method as leave-one-out.
# cv = LeaveOneOut()

# # For each model.
# for model in models:
    
#     # Evalutate models with LOOCV.
#     scores = cross_val_score(model, X, Y, 
#                         scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#     # Calculate MSE.
#     mse = np.mean(np.absolute(scores))
    
#     # Add MSE to overall results table for each model.
#     res.at[models.index(model),'LOOCV_MSE'] = mse

'''
Plot estimated density function for features
'''

# plt.figure(figsize=(18,9))
# for c in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
#     sns.kdeplot(df[c], fill=True,label=c)
# plt.legend()

'''
Display results
'''

# # Force display of full table.
# pd.set_option('display.expand_frame_repr', False)

# # Print overall results table.
# print()
# print(res)

# # Reset display option back to default.
# pd.set_option('display.expand_frame_repr', True)



