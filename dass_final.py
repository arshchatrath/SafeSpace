import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')


RESULTS_DIR = "result_DASS21"
os.makedirs(RESULTS_DIR, exist_ok=True)
EXCEL_FILE = "Final DataSet of DASS21.xlsx"


plt.switch_backend('Agg')


print("📊 Loading DASS-21 dataset...")
df = pd.read_excel(EXCEL_FILE, sheet_name='fyp', engine='openpyxl')
stress_columns = ['q1(S)', 'q6(s)', 'q8(s)', 'q11(s)', 'q12(s)', 'q14(s)', 'q18(s)']
X = df[stress_columns].values
stress_scores = df['Stress'].values * 2


y = np.select([
    stress_scores <= 18,
    (stress_scores > 18) & (stress_scores < 26),
    stress_scores >= 26
], [0, 1, 2])

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")


print("🔄 Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
)


print("⚖ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)


print("🏗 Training Stacking Classifier...")
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=30, max_depth=3, min_samples_split=20, 
        min_samples_leaf=10, max_features='sqrt', bootstrap=True, 
        oob_score=True, random_state=42, class_weight='balanced'
    )),
    ('lr', LogisticRegression(
        C=0.01, penalty='l2', solver='lbfgs', multi_class='multinomial', 
        max_iter=1000, random_state=42, class_weight='balanced'
    )),
    ('nb', GaussianNB())
]

meta_learner = LogisticRegression(
    C=1.0, random_state=42, class_weight='balanced'
)

model = StackingClassifier(
    estimators=base_models, 
    final_estimator=meta_learner, 
    cv=5, 
    stack_method='predict_proba'
)
model.fit(X_train_scaled, y_train)


print("💾 Saving model and scaler...")
joblib.dump(model, os.path.join(RESULTS_DIR, "stacking_classifier_model.pkl"))
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))


print("🔮 Making predictions...")
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)


print("📈 Calculating metrics...")
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(
    y_test, y_pred, 
    target_names=['Low', 'Medium', 'High'], 
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))


print("🔳 Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', 
    xticklabels=['Low', 'Medium', 'High'], 
    yticklabels=['Low', 'Medium', 'High']
)
plt.title("Confusion Matrix - DASS-21 Stress Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print("📊 Generating ROC curves...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ['darkorange', 'darkgreen', 'darkred']
labels = ['Low', 'Medium', 'High']

for i, (color, label) in enumerate(zip(colors, labels)):
    plt.plot(
        fpr[i], tpr[i], color=color, lw=2,
        label=f'{label} (AUC = {roc_auc[i]:.3f})'
    )

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - DASS-21 Stress Classification')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
plt.close()


print("💾 Saving predictions...")
pred_df = pd.DataFrame(y_proba, columns=['prob_Low', 'prob_Medium', 'prob_High'])
pred_df['true_label'] = y_test
pred_df['predicted_label'] = y_pred
pred_df.to_csv(os.path.join(RESULTS_DIR, "test_softmax_outputs.csv"), index=False)

print("📦 Generating softmax outputs for ALL samples...")
all_probs = model.predict_proba(X_all_scaled)
all_preds = model.predict(X_all_scaled)

df_all = pd.DataFrame(all_probs, columns=['prob_Low', 'prob_Medium', 'prob_High'])
df_all['true_label'] = y
df_all['predicted_label'] = all_preds
df_all.to_csv(os.path.join(RESULTS_DIR, "softmax_outputs_ALL.csv"), index=False)



print("\n🔍 Starting XAI Analysis...")


print("🎯 Computing SHAP values...")


def model_predict_proba(X):
    """Wrapper function for SHAP compatibility"""
    return model.predict_proba(X)


def create_background_data(X_train, n_clusters=50):
    """Create background data using sklearn KMeans"""
    n_clusters = min(n_clusters, len(X_train))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    return kmeans.cluster_centers_


print("Creating background data for SHAP...")
background = create_background_data(X_train_scaled, n_clusters=50)


explainer = shap.KernelExplainer(model_predict_proba, background)


test_sample_size = min(30, len(X_test_scaled))  
test_sample_indices = np.random.choice(len(X_test_scaled), test_sample_size, replace=False)
X_test_sample = X_test_scaled[test_sample_indices]
y_test_sample = y_test[test_sample_indices]

print(f"Computing SHAP values for {test_sample_size} test samples...")
try:
    shap_values = explainer.shap_values(X_test_sample, silent=True)
    print("✅ SHAP values computed successfully!")
except Exception as e:
    print(f"❌ SHAP computation failed: {e}")
    print("🔄 Falling back to TreeExplainer for Random Forest...")
    
    
    rf_model = model.named_estimators_['rf']
    tree_explainer = shap.TreeExplainer(rf_model)
    shap_values = tree_explainer.shap_values(X_test_sample)
    print("✅ TreeExplainer SHAP values computed successfully!")


print("📊 Creating SHAP summary plots...")
feature_names = ['Q1(S)', 'Q6(S)', 'Q8(S)', 'Q11(S)', 'Q12(S)', 'Q14(S)', 'Q18(S)']


if isinstance(shap_values, list):
    
    class_names = ['Low', 'Medium', 'High']
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[i], X_test_sample, 
            feature_names=feature_names,
            show=False,
            title=f'SHAP Summary - {class_name} Stress'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_summary_{class_name.lower()}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    
    print("📈 Computing SHAP feature importance...")
    shap_importance = {}
    for i, class_name in enumerate(class_names):
        importance = np.abs(shap_values[i]).mean(axis=0)
        shap_importance[class_name] = importance
else:
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test_sample, 
        feature_names=feature_names,
        show=False,
        title='SHAP Summary - Overall Feature Importance'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_overall.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    
    shap_importance = {}
    overall_importance = np.abs(shap_values).mean(axis=0)
    for class_name in ['Low', 'Medium', 'High']:
        shap_importance[class_name] = overall_importance


importance_df = pd.DataFrame(shap_importance, index=feature_names)
importance_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"))


plt.figure(figsize=(12, 8))
x = np.arange(len(feature_names))
width = 0.25

plt.bar(x - width, importance_df['Low'], width, label='Low Stress', alpha=0.8, color='green')
plt.bar(x, importance_df['Medium'], width, label='Medium Stress', alpha=0.8, color='orange')
plt.bar(x + width, importance_df['High'], width, label='High Stress', alpha=0.8, color='red')

plt.xlabel('DASS-21 Stress Questions')
plt.ylabel('Mean |SHAP Value|')
plt.title('SHAP Feature Importance by Stress Level')
plt.xticks(x, feature_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_feature_importance.png"), dpi=300)
plt.close()


print("🍋 Performing LIME analysis...")


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=['Low', 'Medium', 'High'],
    mode='classification',
    discretize_continuous=True
)


lime_explanations = []
n_lime_samples = min(5, len(X_test_scaled))  

print(f"Generating LIME explanations for {n_lime_samples} samples...")
for i in range(n_lime_samples):
    try:
        explanation = lime_explainer.explain_instance(
            X_test_scaled[i], 
            model.predict_proba, 
            num_features=len(feature_names),
            top_labels=3
        )
        lime_explanations.append(explanation)
        
        
        explanation.save_to_file(
            os.path.join(RESULTS_DIR, f"lime_explanation_sample_{i}.html")
        )
        print(f"✅ LIME explanation {i+1}/{n_lime_samples} completed")
    except Exception as e:
        print(f"❌ LIME explanation {i+1} failed: {e}")
        continue


if lime_explanations:
    print("📋 Creating LIME summary...")
    lime_summary = []
    for i, explanation in enumerate(lime_explanations):
        if i < len(y_pred):
            pred_class = y_pred[i]
            true_class = y_test[i]
            
            
            exp_map = explanation.as_map()[pred_class]
            feature_importance = {feature_names[feat_idx]: importance 
                                for feat_idx, importance in exp_map}
            
            lime_summary.append({
                'sample_id': i,
                'true_class': true_class,
                'predicted_class': pred_class,
                'prediction_probability': y_proba[i][pred_class],
                **feature_importance
            })

    if lime_summary:
        lime_df = pd.DataFrame(lime_summary)
        lime_df.to_csv(os.path.join(RESULTS_DIR, "lime_explanations_summary.csv"), index=False)
        print("✅ LIME summary created successfully!")


print("📊 Creating individual prediction explanations...")


interesting_cases = []
for i in range(min(5, len(y_test))):
    if i < len(y_pred) and y_test[i] != y_pred[i]:  
        interesting_cases.append(i)
    elif len(interesting_cases) < 3:  
        interesting_cases.append(i)

for case_idx in interesting_cases:
    if case_idx < len(X_test_scaled):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        
        feature_values = X_test_scaled[case_idx]
        ax1.barh(feature_names, feature_values, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Standardized Feature Values')
        ax1.set_title(f'Sample {case_idx}: Feature Values\nTrue: {y_test[case_idx]}, Pred: {y_pred[case_idx]}')
        ax1.grid(True, alpha=0.3)
        
        
        probs = y_proba[case_idx]
        colors = ['green', 'orange', 'red']
        bars = ax2.bar(['Low', 'Medium', 'High'], probs, color=colors, alpha=0.7)
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Prediction Probabilities')
        ax2.set_ylim(0, 1)
        
        
        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"individual_explanation_{case_idx}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()


print("📝 Creating interpretability report...")


rf_model = model.named_estimators_['rf']
feature_importance_rf = rf_model.feature_importances_


report_data = {
    'Feature': feature_names,
    'RF_Importance': feature_importance_rf,
    'SHAP_Low': importance_df['Low'].values,
    'SHAP_Medium': importance_df['Medium'].values,
    'SHAP_High': importance_df['High'].values,
    'SHAP_Average': importance_df.mean(axis=1).values
}

interpretability_df = pd.DataFrame(report_data)
interpretability_df = interpretability_df.sort_values('SHAP_Average', ascending=False)
interpretability_df.to_csv(os.path.join(RESULTS_DIR, "interpretability_report.csv"), index=False)


plt.figure(figsize=(14, 8))
x = np.arange(len(feature_names))
width = 0.15

plt.bar(x - 2*width, interpretability_df['RF_Importance'], width, 
        label='Random Forest', alpha=0.8, color='purple')
plt.bar(x - width, interpretability_df['SHAP_Low'], width, 
        label='SHAP Low', alpha=0.8, color='green')
plt.bar(x, interpretability_df['SHAP_Medium'], width, 
        label='SHAP Medium', alpha=0.8, color='orange')
plt.bar(x + width, interpretability_df['SHAP_High'], width, 
        label='SHAP High', alpha=0.8, color='red')
plt.bar(x + 2*width, interpretability_df['SHAP_Average'], width, 
        label='SHAP Average', alpha=0.8, color='black')

plt.xlabel('DASS-21 Stress Questions')
plt.ylabel('Feature Importance')
plt.title('Comprehensive Feature Importance Analysis')
plt.xticks(x, interpretability_df['Feature'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comprehensive_feature_importance.png"), 
            dpi=300, bbox_inches='tight')
plt.close()


print("\n🏆 Final Model Performance and XAI Summary")
print("=" * 60)
print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🎯 Accuracy: {acc:.4f}")
print(f"📈 Weighted F1 Score: {f1:.4f}")
print(f"🔍 XAI Analysis Complete:")
print(f"   - SHAP values computed for {test_sample_size} test samples")
print(f"   - LIME explanations generated for {len(lime_explanations)} samples")
print(f"   - Individual case explanations: {len(interesting_cases)} cases")
print(f"📁 Results saved to: {RESULTS_DIR}/")
print("=" * 60)


summary_data = {
    'Metric': ['Accuracy', 'Weighted F1', 'Test Samples', 'Total Samples'],
    'Value': [acc, f1, len(X_test), len(X)],
    'XAI_Components': ['SHAP Global', 'LIME Local', 'Feature Importance', 'Individual Cases']
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(RESULTS_DIR, "final_summary.csv"), index=False)

print("✅ Script completed successfully!")
print("🔗 Check the result_DASS21 folder for all outputs including XAI visualizations.")
print("\n🔧 Fixes Applied:")
print("   - Replaced shap.kmeans with sklearn KMeans for background data")
print("   - Added TreeExplainer fallback for Random Forest")
print("   - Enhanced error handling for SHAP and LIME")
print("   - Reduced sample sizes for better performance")
print("   - Added robust exception handling throughout XAI section")