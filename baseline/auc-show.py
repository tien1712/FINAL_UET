import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load the result5.csv file
data = pd.read_csv('results/result4.csv')

# Extract predictions and true labels
y_prob = data['prediction']
y_true = data['CHOICE']

# Binarize the output
classes = [1, 2, 3, 4]
y_true_binarized = label_binarize(data['CHOICE'], classes=classes)
y_prob_binarized = label_binarize(data['prediction'], classes=classes)

# Fit the model
model = OneVsRestClassifier(LogisticRegression())
model.fit(data[['prediction']], y_true_binarized)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], data['prediction'] == classes[i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} ROC curve (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Configure plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
plt.legend(loc='lower right')

# Display plot
plt.grid(True)
plt.show()