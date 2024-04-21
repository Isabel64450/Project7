import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

filename="df_sample_frac_cleaned.csv"

df_train=pd.read_csv(filename)

# Initialiser MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


mlflow.set_experiment("Projet7")


df_train= df_train.drop(columns=['SK_ID_CURR'])

df_train, target = df_train.drop(columns="TARGET"), df_train["TARGET"]


categorical_columns_selector = selector(dtype_include=object)
categorical_columns_train = categorical_columns_selector(df_train)
len(categorical_columns_train)


numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns_train = numerical_columns_selector(df_train)
len(numerical_columns_train)




#from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(df_train, target, random_state=42, test_size=0.3,stratify=target)


from imblearn.over_sampling import SMOTE



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



preprocessor.fit(data_train,target_train)



def cross_validate_std(*args, **kwargs):
    """Like cross_validate, except also gives the standard deviation of the score"""
    res = pd.DataFrame(cross_validate(*args, **kwargs))
    res_mean = res.mean()

    res_mean["std_test_score"] = res["test_score"].std()
    if "train_score" in res:
        res_mean["std_train_score"] = res["train_score"].std()
    return res_mean



preprocess_train=preprocessor.fit_transform(data_train)
df_preprocess_train=pd.DataFrame(preprocess_train)


preprocess_test=preprocessor.fit_transform(data_test)
df_preprocess_test=pd.DataFrame(preprocess_test)



target_train.value_counts()



target_test.value_counts()



ore_columns = list(preprocessor.named_transformers_['categorical'].named_steps['OrdianlEncoder'].get_feature_names_out(categorical_columns_train))
new_columns = numerical_columns_train + ore_columns
len(new_columns)



smote = SMOTE(sampling_strategy='minority')



X_resampled, y_resampled = smote.fit_resample(df_preprocess_train,target_train)




X_sm_test, y_sm_test = smote.fit_resample(df_preprocess_test,target_test)



y_resampled.value_counts()



y_sm_test.value_counts()


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
def custom_scorer(y_true, y_pred, probas):
    """
    Custom scoring function that computes precision, recall, F1-score, and AUC.
    Assumes y_true and y_pred are binary classification labels.
    probas: Predicted class probabilities (output of model.predict_proba).
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probas[:, 1])  # Assuming positive class probabilities

    return {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

def find_best_threshold(threshould, fpr, tpr):
   t = threshould[np.argmax(tpr*(1-fpr))]
   # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
   value_t= max(tpr*(1-fpr))
   return value_t


mlflow.set_experiment("Projet 7")

with mlflow.start_run(run_name='DTC nested'):     
       
       modelDTC_N = (DecisionTreeClassifier(max_depth=10))
       proba= modelDTC_N.fit(X_resampled, y_resampled).predict_proba(X_resampled)

       param_grid = {"min_samples_leaf": [1, 10,100], "max_depth": [10,100,1000]}
       

       # Declare the inner and outer cross-validation strategies
       inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
       outer_cv = KFold(n_splits=3, shuffle=True, random_state=0)

       # Inner cross-validation for parameter search
       search_DN = GridSearchCV(estimator=modelDTC_N, param_grid=param_grid,cv=inner_cv, n_jobs=2,scoring=custom_scorer)
       search_DN.fit(X_resampled, y_resampled)
       best_model = search_DN.best_estimator_
       best_params = search_DN.best_params_
       y_pred = best_model.predict(X_sm_test)
       auc = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['auc']
       f1 = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['f1']
       recall = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['recall']
       precision = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['precision']
       
       tn, fp, fn, tp = confusion_matrix(y_sm_test, y_pred).ravel()
       bs = fp + 10 * fn
       
       best_params=search_DN.best_params_   
       mlflow.log_param('best',best_params)
       # Outer cross-validation to compute the testing score
       test_score = cross_val_score(search_DN, X_resampled, y_resampled, cv=outer_cv, n_jobs=2,scoring=custom_scorer)
       mlflow.log_metric("AUC", auc)
       mlflow.log_metric('f1',f1)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_pred)
       #auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search_DN,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       #mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search_DN, X_sm_test,y_sm_test)    
       t=find_best_threshold(thresholds, fpr, tpr)
       mlflow.log_param('threshold',t) 
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search_DN, "model DTC nested",registered_model_name="sk-learn-desition-tree-reg-model")
       # Log the sklearn model and register as version 1
      



with mlflow.start_run(run_name='LR model'):    
       
       modelLR = (LogisticRegression(solver='lbfgs', max_iter=1000))
       proba= modelLR.fit(X_resampled, y_resampled).predict_proba(X_resampled)

       param_grid = {"solver": ['sag', 'saga','lbfgs'], "max_iter": [10,100,1000]}
       
       search = GridSearchCV(estimator=modelLR, param_grid=param_grid, n_jobs=2,scoring=custom_scorer)
       search.fit(X_resampled, y_resampled) 

       best_model = search.best_estimator_
       best_params = search.best_params_
       y_pred = best_model.predict(X_sm_test)
       auc = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['auc']
       f1 = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['f1']
       recall = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['recall']
       precision = custom_scorer(y_sm_test, y_pred, probas=best_model.predict_proba(X_sm_test))['precision']
    

       tn, fp, fn, tp = confusion_matrix(y_sm_test, y_pred).ravel()
       bs = fp + 10 * fn
       best_params=search.best_params_   
       mlflow.log_param('best',best_params)
       mlflow.log_metric("AUC", auc)
       mlflow.log_metric('f1',f1)
       mlflow.log_metric('precision',precision)
       mlflow.log_metric('recall',recall)
       mlflow.log_metric('Bussines score',bs)

       fpr, tpr, thresholds = roc_curve(y_sm_test, y_pred)
       #auc_value = auc(fpr, tpr)
       generate_pr_roc_plots(search,X_sm_test,y_sm_test)
       plt.savefig("roc_curve.png")
       plt.close()
       mlflow.log_artifact("roc_curve.png")
       #mlflow.log_metric('AUC',auc_value)
       ConfusionMatrixDisplay.from_estimator(search, X_sm_test,y_sm_test)    
       t=find_best_threshold(thresholds, fpr, tpr)
       mlflow.log_param('threshold',t) 
       plt.savefig("matrix_confusion.png")
       plt.close()
       mlflow.log_artifact("matrix_confusion.png")
       mlflow.sklearn.log_model(search, "model LR")
# Saving model to disk
pickle.dump(best_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))



















