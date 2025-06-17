
Cheat cheeat

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np


from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
 

## Lire Fichiers

df = pd.read_csv(filepath)
print(f" dataset length : {len(df)}")
## Data previsualisation:

df.head()

df.describe(exclude=[object])  


## DATA CLEANING : 

print(f" length of data set {len(df)}")
df.dropna(inplace=True)
print(f" length of data set {len(df)}")
df.drop_duplicates(inplace=True)
print(f" length of data set {len(df)}")


                        DataFrame.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
                        df.isnull().sum()
                        df.dropna(inplace=True)

                        DataFrame.isna
                        DataFrame.fillna(value=None
                        DataFrame.dropna
                        outlier_indices = np.where((df_diabetics['bmi'] > 0.12) & (df_diabetics['bp'] < 0.8))
                        no_outliers = df_diabetics.drop(outlier_indices[0])



y = df['TARGET_5Yrs']
X = df.drop('Name',  axis=1)
X = X.drop('TARGET_5Yrs', axis=1)


corr = X.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(X.corr(), fignum=f.number,  cmap=cm.get_cmap('magma'))
plt.xticks(range(len(corr.columns)), corr.columns, fontsize=14);
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=14);
cb = plt.colorbar()
cb.ax.tick_params(labelsize=16)
plt.title('Matrice de correlation', fontsize=16);


# Column/ variable selection
col = ['GP','PTS', 'FG%','3P Made', '3P%', 'FT%', 'REB','AST']
X0 = X[col]

## Plotting :

# plot data
plot_hists(X0, 3, 3, list(X0.columns))


## Normalizing:

X_norm = X[cols].copy()

power1          = PowerTransformer(method='yeo-johnson', standardize=True)
X_norm[['GP']]  = power1.fit_transform(X_norm[['GP']].values.reshape(-1, 1))

std_sc              = StandardScaler()
X_norm[['3P Made']] = std_sc.fit_transform(X_norm[['3P Made']].values.reshape(-1, 1))

## Revisualize after normalization.
plot_hists(X_norm, 3, 3, list(X0.columns))

# PCA 
from sklearn.decomposition import PCA
X   = X.fillna(0)
n   = len(X.columns)
pca = PCA(n_components=n)
pca.fit(X)

fig, ax = plt.subplots(figsize=(10, 3))
ax.scatter(np.arange(n), pca.explained_variance_ratio_)
ax.set_xticks(np.arange(n))
ax.set_xticklabels(np.arange(n), fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14);
ax.set_ylabel('% explained variance', fontsize=14);
ax.set_title('PCA', fontsize=14);

# TSNEE
# Apply t-SNE to the data
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)


## Class equilibrium:

labels, counts = np.unique(y.to_numpy(), return_counts=True)
plt.figure(figsize=(8,3)) 
plt.bar(labels, counts, align='center', facecolor = '#2ab0ff', width=0.4)
plt.gca().set_xticks(labels)
plt.gca().set_xticklabels(['carrière: < 5 ans','carrière:  ≥ 5 ans'], fontsize=14)
plt.gca().set_ylabel('# de joueurs débutants', fontsize=14)
plt.xlim(-0.6, 1.6)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


## Base model

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)
y_pred = dummy_clf.predict(X)
print('Valeurs exactes: ')
print('Recall: ' + str(recall_score(y, y_pred)))
print('Precision: ' + str(precision_score(y, y_pred)))
print('F1-score: ' + str(f1_score(y, y_pred)))



## 

# ML train/test split

a)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
b) Train/validation/test
# Assuming X and y are your features and labels
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
 


# Logistic regression

param_grid = {'penalty': ['elasticnet'],
              'C':[0.001, 0.05, 0.08],                   # 0.01, 0.05, 0.1 , 1, 10
              'class_weight': ['balanced',None],
              'l1_ratio': [ 0.7, 0.9, 1],          # 0, 0.1, 0.3, 0.5,
              'max_iter': [5, 7, 10, 20],
              'solver':['saga']
             }
model_to_tune = LogisticRegression()





clf = LogisticRegression()
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)


# SVM

param_grid = {"C": [ 0.1, 1, 10, 100],       # 0.01, 100
               'kernel':['linear', 'rbf']},  #"gamma": 'scale', [0.001, 0.01, 0.1, 1]
                                             # ,'poly','sigmoid'
                                             # 'degree': [3, 5, 10]
model_to_tune = SVC()

# Random forest

param_grid ={'bootstrap': [True, False],
             'max_depth': [53],                 # 20, 100
             'max_features': ['auto', 'sqrt'], 
             'min_samples_leaf': [  15, 23],    # 30, 50, 100 
             'min_samples_split': [ 50],        # 10, 100, 40
             'n_estimators': [20]}              # 30, 40, 100 
model_to_tune = RandomForestClassifier()


# HGB

param_grid = {'loss':['log_loss'],
              'learning_rate':[0.1],
              'max_iter': [50],       # , 100, 200
              'max_leaf_nodes': [ 30, 100], # None, 5,
              'max_depth':[ 10, 40],    # None, 3,
              #'min_samples_leaf': [5, 20, 100],
              #'l2_regularization':[0, 0.01, 0.1, 1, 10],
              'max_bins':[255],
             }
model_to_tune = HistGradientBoostingClassifier()


# LGBMClassifier

from lightgbm import LGBMClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}
lgbm_model = LGBMClassifier()
grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_



# Create a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

accuracy = grid_search.score(X_test, y_test) # eval on test set

# KNN
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)





# Poids des dans la décision
fig, ax = plt.subplots(figsize=(10, 3))
ax.scatter(np.linspace(0,5,6), model_lr.best_estimator_.coef_)
ax.set_xticks((0, 1, 2, 3, 4, 5))
ax.set_xticklabels(tuple(cols), fontsize=18)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
ax.set_title('Weights of different variables on the decision', fontsize=18)
ax.set_ylabel('Weights', fontsize=18)



## Fontions prêtes


def plot_images(images, l, c):
    """
    Plots m images arranged in l lines and c columns.
    The remaining plots are not displayed.

    Args:
        images (list): List of images.
        l (int): Number of lines.
        c (int): Number of columns.
    """
    M = len(images)
    total_plots = l * c


    fig, axs = plt.subplots(l, c, figsize=(10, 10))
    fig.tight_layout()

    for i in range(l):
        for j in range(c):
            idx = i * c + j
            if idx < m:
                axs[i, j].imshow(images[idx]) # .hist(data[idx], bins=20, alpha=0.7, color="skyblue")
                axs[i, j].axis("off")
            else:
                axs[i, j].axis("off")

    plt.show()

def plot_hists(df, l, c, cols):
    """
    Plots m histograms arranged in l lines and c columns.
    """
    m = len(cols)

    fig, axs = plt.subplots(l, c, figsize=(10, 10))
    fig.tight_layout()

    for i in range(l):
        for j in range(c):
            idx = i * c + j
            if idx < m:
                axs[i, j].hist(df[cols[idx]])
                #axs[i, j].set_yaxis("off")
                axs[i, j].set_title(cols[idx])          
            else:
                axs[i, j].axis("off")
    plt.show()





## Example of K-means clustering:

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)  # Choose K=3
kmeans.fit(points)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_


plt.figure(figsize=(8, 6))
plt.scatter(x=points[:, 0], y=points[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title(f"K-Means Clustering with {n_clusters} Clusters")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.show()


## Confusion matrix

from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

cm = confusion_matrix(y_true, y_pred)

import seaborn as sns

sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
