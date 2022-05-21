# from google.cloud import bigquery
# client = bigquery.Client()
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# import numpy as np
# import pandas as pd
# import itertools
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import  confusion_matrix, recall_score,precision_score, precision_recall_curve, f1_score, fbeta_score
# from sklearn.utils.fixes import signature
# from datsource_utils import get_source_query
#
#
# sql = get_source_query()
# df = client.query(sql).to_dataframe()
# #drop columns with null values
# df.drop(labels=['stddev_output_idle_time','stddev_input_idle_time'], axis=1, inplace=True)
# #get rid of non-numeric features
# features = df.drop(labels=['is_miner','address'], axis=1)
# target = df['is_miner'].values
# indices = range(len(features))
#
# #Train test split
# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, target, indices,  test_size=0.2)
#
# rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
# rf.fit(X_train, y_train)
#
#
# y_pred = rf.predict(X_test) #
# probs = rf.predict_proba(X_test)[:,1] #positive class probabilities
#
#
# precision, recall, thresholds = precision_recall_curve(y_test, probs)
# print(f"precision:{precision}, recall:{recall}")
#
# # Precision / recall curve code adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
#
# fig, ax = plt.subplots(figsize=(8,6))
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.0])
# plt.xlim([0.0, 1.0])
# ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# plt.title('Mining Pool Detector - Precision/Recall Curve', fontsize=14)
#
#
# #confusion matrix code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#     dummy=np.array([[0,0],[0,0]])
#     plt.figure(figsize=(8,6))
#     plt.imshow(dummy, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# class_names = ['not mining pool', 'mining pool']
# np.set_printoptions(precision=2)
#
# # Plot confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
#                       title='Mining Pool Detector - Confusion Matrix')
#
# plt.show()
#
# ############## plot feature important
# x_pos = np.arange(len(features.columns))
# btc_importances = rf.feature_importances_
#
# inds = np.argsort(btc_importances)[::-1]
# btc_importances = btc_importances[inds]
# cols = features.columns[inds]
# bar_width = .8
#
# #how many features to plot?
# n_features=12
# x_pos = x_pos[:n_features][::-1]
# btc_importances = btc_importances[:n_features]
#
# #plot
# plt.figure(figsize=(12,6))
# plt.barh(x_pos, btc_importances, bar_width, label='BTC model')
# plt.yticks(x_pos, cols, rotation=0, fontsize=14)
# plt.xlabel('feature importance', fontsize=14)
# plt.title('Mining Pool Detector', fontsize=20)
# plt.tight_layout()
#
# #Are False Positives associated with dark mining pools?
#
# #data points where model predicts true, but are labelled as false
# false_positives = (y_test==False) & (y_pred==True)
#
# #subset to test set data only
# df_test = df.iloc[indices_test, :]
#
# print('False Positive addresses')
#
# #subset test set to false positives only
# df_test.iloc[false_positives].head(15)