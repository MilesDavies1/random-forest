import scipy
import pickle
import xgboost
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score

universal_no_outliers = r'D:/users/Mmile/Documents/CS/universal_no_outliers.csv'
universal = r'D:/users/Mmile/Documents/CS/universal_combined_training_set_final.csv'
english_no_outliers = r'D:/users/Mmile/Documents/CS/english_no_outliers.csv'
english = r'D:/users/Mmile/Documents/CS/english_combined_training_set_final.csv'

def turn_orgs_to_bots(df):  
  df = df.replace({'labels': 2}, {'labels': 1})
  return df

def fix_positive_condition(df):
  df = df.replace({'labels': 1}, {'labels': 3})
  df = df.replace({'labels': 0}, {'labels': 1})
  df = df.replace({'labels': 3}, {'labels': 0})
  return df

def remove_outliers(zscore, pathA, pathB):  
  mdf = pd.read_csv(pathA)
  mdf = fix_positive_condition(mdf)  
  mdf = turn_orgs_to_bots(mdf)
  columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer", "labels"]  
  mdf= mdf[columns]
    # separate them by category  
  human_df = mdf[mdf["labels"] == 1]  
  non_human_df = mdf[mdf["labels"] == 0]  
  columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"]

    # for humans
  human_df = human_df[np.abs(stats.zscore(human_df[columns]) < zscore).all(axis=1)]  
  
    # for non_humans  
  non_human_df = non_human_df[np.abs(stats.zscore(non_human_df[columns]) < zscore).all(axis=1)]  
  
    # combine both of the dataframe and export  
  master = pd.concat([human_df, non_human_df])  
  
  master.to_csv(pathB, index=False)

def random_forest(path, lang):
  master_df = pd.read_csv(path)
  x1 = master_df.drop(['labels'], axis=1).values
  y1 = master_df['labels'].values

  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100, stratify=y1)

  X_train_scaled = preprocessing.scale(X_train)
  X_test_scaled = preprocessing.scale(X_test)
  model = xgboost.XGBClassifier()
  optimization_dict = {'max_depth': [2, 4, 6], 'n_estimators': [50, 200, 500], 'learning_rate': [0.1, 0.01, 1]}
  model = model_selection.GridSearchCV(model, optimization_dict, scoring='accuracy', verbose=1)
  model.fit(X_train_scaled, Y_train)

  # print(model.best_score_)
  # print(model.best_params_)

  title_options = [("Confusion Matrix, without normalization", None), ("Normalization: true", "true"), 
                    ("Normalization: pred", "pred"), ("Normalization: all", "all")]
  for title, normalize in title_options:
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, Y_test, display_labels=["bot", "human"], 
                                          cmap=plt.cm.Blues, normalize=normalize)
    file_path = r'D:/users/Mmile/Documents/file-{}-{}.png'.format(normalize, lang)
    disp.ax_.set_title(title)
    plt.savefig(file_path)

# confusion matrix figures for english accounts
remove_outliers(0.9516932907348243, english, english_no_outliers)
random_forest(english_no_outliers, "eng")

# confusion matrix figures for non english accounts
remove_outliers(9129954376960365, universal, universal_no_outliers)
random_forest(universal_no_outliers, "univ")