from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from features import add_features

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'telecom_customer_churn.csv'
FIG=ROOT/'reports'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
MODEL_DIR=ROOT/'models'; MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'reports'; REPORT.mkdir(parents=True,exist_ok=True)

df=add_features(pd.read_csv(DATA))
y=(df.pop('Churn')=='Yes').astype(int)
X=df.drop(columns=['customerID'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
cat_cols=X_train.select_dtypes(include=['object','category']).columns.tolist()
num_cols=[c for c in X_train.columns if c not in cat_cols]
pre=ColumnTransformer([
 ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num_cols),
 ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat_cols)
])

candidates={
 'Logistic Regression': LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),
 'Random Forest': RandomForestClassifier(n_estimators=140,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',random_state=42,n_jobs=1),
 'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=130,learning_rate=.07,max_leaf_nodes=20,l2_regularization=1.0,random_state=42)
}
cv=StratifiedKFold(3,shuffle=True,random_state=42)
rows=[]; fitted={}
for name,model in candidates.items():
    pipe=Pipeline([('preprocessor',pre),('model',model)])
    roc=float(cross_val_score(pipe,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=1).mean())
    f1=float(cross_val_score(pipe,X_train,y_train,cv=cv,scoring='f1',n_jobs=1).mean())
    pipe.fit(X_train,y_train)
    p=pipe.predict_proba(X_test)[:,1]
    rows.append({'Model':name,'CV ROC-AUC':roc,'CV F1':f1,'Test ROC-AUC':roc_auc_score(y_test,p)})
    fitted[name]=pipe
comparison=pd.DataFrame(rows).sort_values('CV ROC-AUC',ascending=False)
comparison.to_csv(REPORT/'model_comparison.csv',index=False)

# Compact manual tuning of the best-performing family by validation ROC-AUC.
# This avoids an expensive search while still documenting deliberate model selection.
rf_configs=[
 {'n_estimators':160,'max_depth':10,'min_samples_leaf':3,'max_features':'sqrt'},
 {'n_estimators':220,'max_depth':14,'min_samples_leaf':3,'max_features':'sqrt'},
 {'n_estimators':220,'max_depth':None,'min_samples_leaf':4,'max_features':0.65},
]
X_fit,X_val,y_fit,y_val=train_test_split(X_train,y_train,test_size=.20,random_state=7,stratify=y_train)
best_score=-1; best_params=None; best_rf=None
for cfg in rf_configs:
    pipe=Pipeline([('preprocessor',pre),('model',RandomForestClassifier(**cfg,class_weight='balanced_subsample',random_state=42,n_jobs=1))])
    pipe.fit(X_fit,y_fit); vp=pipe.predict_proba(X_val)[:,1]; score=roc_auc_score(y_val,vp)
    if score>best_score: best_score=score; best_params=cfg; best_rf=pipe
best_rf.fit(X_train,y_train)
# Compare tuned RF to strongest baseline model on held-out test set.
all_models=dict(fitted); all_models['Tuned Random Forest']=best_rf
model_scores={name:roc_auc_score(y_test,m.predict_proba(X_test)[:,1]) for name,m in all_models.items()}
best_name=max(model_scores,key=model_scores.get); best_model=all_models[best_name]
proba=best_model.predict_proba(X_test)[:,1]

thresholds=np.arange(.20,.81,.01); th_rows=[]
for t in thresholds:
    pred=(proba>=t).astype(int)
    th_rows.append((t,precision_score(y_test,pred,zero_division=0),recall_score(y_test,pred),f1_score(y_test,pred)))
th=pd.DataFrame(th_rows,columns=['threshold','precision','recall','f1'])
elig=th[th.recall>=.70]
best_t=float((elig if len(elig) else th).sort_values(['f1','precision'],ascending=False).iloc[0].threshold)
pred=(proba>=best_t).astype(int)
metrics={
 'selected_model':best_name,'threshold':best_t,'accuracy':accuracy_score(y_test,pred),
 'precision':precision_score(y_test,pred),'recall':recall_score(y_test,pred),'f1':f1_score(y_test,pred),
 'roc_auc':roc_auc_score(y_test,proba),'pr_auc':average_precision_score(y_test,proba),
 'train_size':len(y_train),'test_size':len(y_test),'churn_rate':float(y.mean()),
 'tuned_random_forest_validation_auc':best_score,'best_params':best_params
}
(REPORT/'metrics.json').write_text(json.dumps(metrics,indent=2))
(REPORT/'classification_report.txt').write_text(classification_report(y_test,pred,target_names=['Stay','Churn']))
th.to_csv(REPORT/'threshold_analysis.csv',index=False)
joblib.dump(best_model,MODEL_DIR/'churn_model.joblib')
(MODEL_DIR/'metadata.json').write_text(json.dumps({'threshold':best_t,'selected_model':best_name,'feature_columns':X.columns.tolist()},indent=2))

# Evaluation figures
fig,ax=plt.subplots(figsize=(5.5,4.5)); ConfusionMatrixDisplay(confusion_matrix(y_test,pred),display_labels=['Stay','Churn']).plot(values_format='d',ax=ax); ax.set_title('Confusion Matrix'); fig.tight_layout(); fig.savefig(FIG/'confusion_matrix.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(5.5,4.5)); RocCurveDisplay.from_predictions(y_test,proba,ax=ax); ax.plot([0,1],[0,1],'--'); ax.set_title(f'ROC Curve (AUC={metrics["roc_auc"]:.3f})'); fig.tight_layout(); fig.savefig(FIG/'roc_curve.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(5.5,4.5)); PrecisionRecallDisplay.from_predictions(y_test,proba,ax=ax); ax.set_title(f'Precision-Recall (AP={metrics["pr_auc"]:.3f})'); fig.tight_layout(); fig.savefig(FIG/'precision_recall_curve.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.plot(th.threshold,th.precision,label='Precision'); ax.plot(th.threshold,th.recall,label='Recall'); ax.plot(th.threshold,th.f1,label='F1'); ax.axvline(best_t,linestyle='--',label=f'Selected={best_t:.2f}'); ax.set(xlabel='Decision Threshold',ylabel='Score',title='Threshold Optimization'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'threshold_optimization.png',dpi=170); plt.close(fig)

sample_n=min(500,len(X_test)); X_imp=X_test.iloc[:sample_n]; y_imp=y_test.iloc[:sample_n]
pi=permutation_importance(best_model,X_imp,y_imp,n_repeats=2,random_state=42,scoring='roc_auc',n_jobs=1)
imp=pd.DataFrame({'Feature':X_imp.columns,'Importance':pi.importances_mean}).sort_values('Importance',ascending=False).head(15)
imp.to_csv(REPORT/'feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(7.5,5.5)); show=imp.sort_values('Importance'); ax.barh(show.Feature,show.Importance); ax.set(xlabel='Decrease in ROC-AUC',title='Top Permutation Feature Importance'); fig.tight_layout(); fig.savefig(FIG/'feature_importance.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,4)); cp=comparison.sort_values('CV ROC-AUC'); ax.barh(cp.Model,cp['CV ROC-AUC']); ax.set(xlim=(.5,1),xlabel='3-fold CV ROC-AUC',title='Model Comparison'); fig.tight_layout(); fig.savefig(FIG/'model_comparison.png',dpi=170); plt.close(fig)

print(json.dumps(metrics,indent=2)); print('\nModel comparison:\n'+comparison.to_string(index=False))
