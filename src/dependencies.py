# Data Manipulation and Analysis
import pandas as pd
import numpy as np
import math

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from termcolor import colored as cl # text customization

# Statistical Analysis and Modeling
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from scipy.stats import chi2
from scipy.stats import linregress
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
import benford as bf
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss

# Machine Learning Algorithms
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier

# Outlier Detection
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from PyNomaly import loop
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.pca import PCA as PyOD_PCA
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.xgbod import XGBOD
from pyod.models.gmm import GMM

# Performance Evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Time Series Analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima

# Miscellaneous
from collections import Counter
import itertools
from functools import reduce
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, HTML
import shap
import lime
import warnings
warnings.filterwarnings('ignore')

# Insert the parent path relative to this notebook so we can import from the src folder.
import sys
sys.path.insert(0, "..")

# Applying style to graphs
#plt.style.use('ggplot')

# Display settings
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.options.display.float_format = '{:.2f}'.format
