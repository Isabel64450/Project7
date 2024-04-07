#!/usr/bin/env python
# coding: utf-8

# # Implémentez un modèle de scoring

# 
# # Sommaire :
# 
# **Partie 1 : Importation et preparation des données**
#     
#  - <a href="#C1">Importation des données nettoyées</a>
#  - <a href="#C2">Choix de cible et séparation des données en train and test</a>
# 
# 
# **Partie 2 : Modelisation des données**
#     
#  - <a href="#C3">Multiple modélisation avec différents classifiers</a>
#  - <a href="#C4">Évaluation des modèles </a>
#  - <a href="#C5">Tuning hyperparamètres</a>
#  - <a href="#C6">Predictions avec le modèle choisi</a>
#  - <a href="#C7">Importance des  variables</a>
#  
#     
# **Partie 3 : Conclusion et observations**
#  
#  - <a href="#C9">Conclusion</a>
#  - <a href="#C10">Observations</a>
#  
#     

# In[1]:


import mlflow
import os
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
#from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay


# In[2]:


path="./DataP7/Cleaned/"
filename="df_app_train_cleaned.csv"


# In[3]:


df_train=pd.read_csv(path + filename)
df_train.head()


# In[4]:


filename1="df_app_test_cleaned.csv"


# In[5]:


# Initialiser MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# In[6]:


mlflow.set_experiment("Projet7")


# In[7]:


df_test=pd.read_csv(path + filename1)
df_test.head()


# In[8]:


df_train= df_train.drop(columns=['SK_ID_CURR'])
df_test=df_test.drop(columns=['SK_ID_CURR'])


# In[9]:


df_train, target = df_train.drop(columns="TARGET"), df_train["TARGET"]


# In[10]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns_train = categorical_columns_selector(df_train)
len(categorical_columns_train)


# In[11]:


#categorical_columns_selector = selector(dtype_include=object)
categorical_columns_test = categorical_columns_selector(df_test)
len(categorical_columns_test)


# In[12]:


numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns_train = numerical_columns_selector(df_train)
len(numerical_columns_train)


# In[13]:


numerical_columns_test = numerical_columns_selector(df_test)
len(numerical_columns_test)


# In[14]:


print(
    f"Nombre d’échantillons dans les essais: {df_test.shape[0]} => "
    f"{df_test.shape[0] /( df_train.shape[0] + df_test.shape[0]) * 100:.1f}% du"
    " ensemble original"
)


# In[15]:


print(
    f"Nombre d’échantillons dans les essais: {df_train.shape[0]} => "
    f"{df_train.shape[0] /( df_train.shape[0] + df_test.shape[0]) * 100:.1f}% du"
    " ensemble original"
)


# In[16]:


#from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(df_train, target, random_state=42, test_size=0.3,stratify=target)


# In[17]:


from imblearn.over_sampling import SMOTE


# In[20]:


numerique_transformer = Pipeline(steps=[("imputer", SimpleImputer(missing_values=np.nan, fill_value=0)),
        ("StandarScaler",StandardScaler()),])




categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="constant", fill_value='nada')),
                                        ("OrdianlEncoder",OrdinalEncoder()),])
                           

categorical_transformer1=Pipeline(steps=[('imputer',SimpleImputer(missing_values=np.nan, fill_value=0)),
                                         ('OneHotEncoder',OneHotEncoder())])


preprocessor=ColumnTransformer(transformers=[('numerical',numerique_transformer,numerical_columns_train), 
                                              ('categorical',categorical_transformer,categorical_columns_train)],remainder ='passthrough')
                                            
preprocessor1=ColumnTransformer(transformers=[('numerical',numerique_transformer,numerical_columns_train),
                                              ('categorical',categorical_transformer,categorical_columns_train)])


# In[ ]:





# In[22]:


preprocessor.fit(data_train,target_train)


# In[ ]:





# In[23]:


def cross_validate_std(*args, **kwargs):
    """Like cross_validate, except also gives the standard deviation of the score"""
    res = pd.DataFrame(cross_validate(*args, **kwargs))
    res_mean = res.mean()

    res_mean["std_test_score"] = res["test_score"].std()
    if "train_score" in res:
        res_mean["std_train_score"] = res["train_score"].std()
    return res_mean


# In[ ]:





# In[26]:


preprocess_train=preprocessor.fit_transform(data_train)
df_preprocess_train=pd.DataFrame(preprocess_train)


# In[27]:


preprocess_test=preprocessor.fit_transform(data_test)
df_preprocess_test=pd.DataFrame(preprocess_test)


# In[28]:


target_train.value_counts()


# In[29]:


target_test.value_counts()


# In[30]:


ore_columns = list(preprocessor.named_transformers_['categorical'].named_steps['OrdianlEncoder'].get_feature_names_out(categorical_columns_train))
new_columns = numerical_columns_train + ore_columns
len(new_columns)


# In[31]:


smote = SMOTE(sampling_strategy='minority')


# In[32]:


X_resampled, y_resampled = smote.fit_resample(df_preprocess_train,target_train)


# In[33]:


X_sm_test, y_sm_test = smote.fit_resample(df_preprocess_test,target_test)


# In[34]:


y_resampled.value_counts()


# In[35]:


y_sm_test.value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


modelBRC = make_pipeline(preprocessor,BalancedRandomForestClassifier(n_estimators=10, random_state=42))
modelBRC.fit(data_train,target_train)


# In[40]:


def generate_pr_roc_plots(model, data_test, target_test):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(
        model,
        data_test,
        target_test,
        marker="+",
        plot_chance_level=True,
        chance_level_kw={"color": "tab:orange", "linestyle": "--"},
        ax=axs[0],
    )

    # ROC Curve
    RocCurveDisplay.from_estimator(
        model,
        data_test,
        target_test,
        marker="+",
        plot_chance_level=True,
        chance_level_kw={"color": "tab:orange", "linestyle": "--"},
        ax=axs[1],
    )

    _ = fig.suptitle("PR and ROC curves")

# Example usage:
# generate_pr_roc_plots(your_search, your_X_sm_test, your_y_sm_test)


# In[41]:


mlflow.set_experiment("Experiment-4-P7")


# In[42]:


with mlflow.start_run(run_name='BRC not nested'):    
       
       modelBRC = make_pipeline(preprocessor,BalancedRandomForestClassifier(n_estimators=10, random_state=42))
       modelBRC.fit(data_train,target_train)


       param_grid = {"balancedrandomforestclassifier__n_estimators": [0.1, 1, 10,100], "balancedrandomforestclassifier__max_depth": [1, 10,100,1000]}
       
       search = GridSearchCV(estimator=modelBRC, param_grid=param_grid, n_jobs=2,scoring='roc_auc')
       search.fit(data_train,target_train)
       target_predicted = search.predict(data_test)
       accuracy = accuracy_score(target_test, target_predicted)
       precision = precision_score(target_test, target_predicted)
       recall = recall_score(target_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(target_test, target_predicted).ravel()
       bs = fp + 10 * fn
       best=search.best_params_   
       mlflow.log_param('best',best)
       mlflow.log_metric("score of the best", search.best_score_)
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       
       y_preds = search.predict(data_test).ravel()

       fpr, tpr, thresholds = roc_curve(target_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search,data_test,target_test)
    
                  
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search, data_test, target_test,cmap='magma')
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search, "model BRC")



      
      


# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


with mlflow.start_run(run_name='Dummy not nested'):    
       
       modelDummy = (DummyClassifier(strategy='most_frequent'))
       modelDummy.fit(X_resampled, y_resampled)


       param_grid = {"strategy": ['most_frequent']}
       
       search_Dummy = GridSearchCV(estimator=modelDummy, param_grid=param_grid, n_jobs=2,scoring='roc_auc')
       search_Dummy.fit(X_resampled, y_resampled)  
       target_predicted = search_Dummy.predict(X_sm_test)
       accuracy = accuracy_score(y_sm_test, target_predicted)
       precision = precision_score(y_sm_test, target_predicted)
       recall = recall_score(y_sm_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(y_sm_test, target_predicted).ravel()
       bs = fp + 10 * fn
       best=search_Dummy.best_params_   
       mlflow.log_param('best',best)
       mlflow.log_metric("score of the best", search_Dummy.best_score_)
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       y_preds = search_Dummy.predict(X_sm_test).ravel()

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search_Dummy,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search_Dummy, X_sm_test,y_sm_test,cmap='cividis')    
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search_Dummy, "model Dummy not nested")

  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


with mlflow.start_run(run_name='BRC nested'):     
       
       modelBRC_N = make_pipeline(preprocessor,BalancedRandomForestClassifier(n_estimators=10, random_state=42))
       modelBRC_N.fit(data_train,target_train)


       param_grid = {"balancedrandomforestclassifier__n_estimators": [1, 10,100], "balancedrandomforestclassifier__max_depth": [10,100,1000]}
           
       # Declare the inner and outer cross-validation strategies
       inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
       outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

       # Inner cross-validation for parameter search
       search_N = GridSearchCV(estimator=modelBRC_N, param_grid=param_grid,cv=inner_cv, n_jobs=2,scoring='roc_auc')
       search_N.fit(data_train,target_train)
       target_predicted = search_N.predict(data_test)
       accuracy = accuracy_score(target_test, target_predicted)
       precision = precision_score(target_test, target_predicted)
       recall = recall_score(target_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(target_test, target_predicted).ravel()
       bs = fp + 10 * fn
       
       best=search.best_params_   
       mlflow.log_param('best',best)
       # Outer cross-validation to compute the testing score
       test_score = cross_val_score(search_N, data_train, target_train, cv=outer_cv, n_jobs=2,scoring='roc_auc')
       mlflow.log_metric("score of the best", test_score.mean())
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       y_preds = search.predict(data_test).ravel()

       fpr, tpr, thresholds = roc_curve(target_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search_N,data_test,target_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search_N, data_test,target_test)    
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search_N, "model BRC nested")




    
           
       
       


# In[46]:


with mlflow.start_run(run_name='DTC not nested'):    
       
       modelDTC = (DecisionTreeClassifier(max_depth=5))
       modelDTC.fit(X_resampled, y_resampled)


       param_grid = {"min_samples_leaf": [0.1, 1, 10,100], "max_depth": [1, 10,100,1000]}
       
       search = GridSearchCV(estimator=modelDTC, param_grid=param_grid, n_jobs=2,scoring='roc_auc')
       search.fit(X_resampled, y_resampled)  
       target_predicted = search.predict(X_sm_test)
       accuracy = accuracy_score(y_sm_test, target_predicted)
       precision = precision_score(y_sm_test, target_predicted)
       recall = recall_score(y_sm_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(y_sm_test, target_predicted).ravel()
       bs = fp + 10 * fn
       best=search.best_params_   
       mlflow.log_param('best',best)
       mlflow.log_metric("score of the best", search.best_score_)
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       y_preds = search.predict(X_sm_test).ravel()

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search, X_sm_test,y_sm_test)    
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search, "model DTC  not nested")


# In[ ]:





# In[ ]:





# In[48]:


with mlflow.start_run(run_name='DTC nested'):     
       
       modelDTC_N = (DecisionTreeClassifier(max_depth=10))
       modelDTC_N.fit(X_resampled, y_resampled)


       param_grid = {"min_samples_leaf": [0.1, 1, 10,100], "max_depth": [1, 10,100,1000]}
    
       

       # Declare the inner and outer cross-validation strategies
       inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
       outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

       # Inner cross-validation for parameter search
       search_DN = GridSearchCV(estimator=modelDTC_N, param_grid=param_grid,cv=inner_cv, n_jobs=2,scoring='roc_auc')
       search_DN.fit(X_resampled, y_resampled)
       target_predicted = search_DN.predict(X_sm_test)
       accuracy = accuracy_score(y_sm_test, target_predicted)
       precision = precision_score(y_sm_test, target_predicted)
       recall = recall_score(y_sm_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(y_sm_test, target_predicted).ravel()
       bs = fp + 10 * fn
       
       best=search.best_params_   
       mlflow.log_param('best',best)
       # Outer cross-validation to compute the testing score
       test_score = cross_val_score(search_DN, X_resampled, y_resampled, cv=outer_cv, n_jobs=2,scoring='roc_auc')
       mlflow.log_metric("score of the best", test_score.mean())
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       y_preds = search_DN.predict(X_sm_test).ravel()

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search_DN,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search_DN, X_sm_test,y_sm_test)    
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search_DN, "model DTC nested")


# In[49]:


with mlflow.start_run(run_name='LR not nested'):    
       
       modelLR = (LogisticRegression(solver='lbfgs', max_iter=1000))
       modelLR.fit(X_resampled, y_resampled)


       param_grid = {"solver": ['sag', 'saga','lbfgs'], "max_iter": [10,100,1000]}
       
       search = GridSearchCV(estimator=modelLR, param_grid=param_grid, n_jobs=2,scoring='roc_auc')
       search.fit(X_resampled, y_resampled)  
       target_predicted = search.predict(X_sm_test)
       accuracy = accuracy_score(y_sm_test, target_predicted)
       precision = precision_score(y_sm_test, target_predicted)
       recall = recall_score(y_sm_test, target_predicted)
       tn, fp, fn, tp = confusion_matrix(y_sm_test, target_predicted).ravel()
       bs = fp + 10 * fn
       best=search.best_params_   
       mlflow.log_param('best',best)
       mlflow.log_metric("score of the best", search.best_score_)
       mlflow.log_metric('accuracy',accuracy)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)
       y_preds = search.predict(X_sm_test).ravel()

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_preds)
       auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search, X_sm_test,y_sm_test)    
       
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search, "model LR not nested")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




