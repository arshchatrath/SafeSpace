import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from zipfile import ZipFile
import pickle
import warnings
warnings.filterwarnings("ignore")


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class PhysioDominantFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weights=None):
        self.class_weights = class_weights if class_weights else {0:1.0, 1:1.0, 2:1.0}
        
        self.mod_weights = {'phys':0.60, 'text':0.25, 'voice':0.15}
        self.feature_names = ['phys_low', 'phys_medium', 'phys_high', 
                             'text_low', 'text_medium', 'text_high',
                             'voice_low', 'voice_medium', 'voice_high']
        self.is_fitted_ = False
        
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
        
    def _check_inputs(self, mod_probs):
        
        if not mod_probs:
            raise ValueError("No modality probabilities provided")
        for mod in mod_probs:
            if not isinstance(mod_probs[mod], np.ndarray):
                mod_probs[mod] = np.array(mod_probs[mod])
            if mod_probs[mod].shape != (3,):
                raise ValueError(f"{mod} probabilities must be length 3")

    def _convert_to_feature_vector(self, mod_probs):
        features = np.zeros(9)  # 3 modalities × 3 classes
        
        if 'phys' in mod_probs:
            features[0:3] = mod_probs['phys']
        if 'text' in mod_probs:
            features[3:6] = mod_probs['text']
        if 'voice' in mod_probs:
            features[6:9] = mod_probs['voice']
            
        return features

    def predict_proba(self, mod_probs):
        
        self._check_inputs(mod_probs)
        
       
        weighted = []
        for mod in ['phys', 'text', 'voice']:
            if mod in mod_probs:
                conf = np.max(mod_probs[mod])  
                weighted.append(mod_probs[mod] * conf * self.mod_weights[mod])
        
        if not weighted:
            return np.array([0.33, 0.33, 0.33])  # Neutral if no data
        
        fused = sum(weighted)
        return fused / fused.sum()  # Normalize

    def predict_proba_from_features(self, X):
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        results = []
        for row in X:
            mod_probs = {}
            if np.any(row[0:3] != 0):
                mod_probs['phys'] = row[0:3]
            if np.any(row[3:6] != 0):
                mod_probs['text'] = row[3:6]
            if np.any(row[6:9] != 0):
                mod_probs['voice'] = row[6:9]
            
            if mod_probs:
                proba = self.predict_proba(mod_probs)
            else:
                proba = np.array([0.33, 0.33, 0.33])
            results.append(proba)
        
        return np.array(results)

    def predict(self, X, true_label=None):
        
        if isinstance(X, dict):
            
            proba = self.predict_proba(X)
            if true_label is not None:
                proba = proba * self.class_weights.get(true_label, 1.0)
            return np.argmax(proba)
        else:
            
            probas = self.predict_proba_from_features(X)
            return np.argmax(probas, axis=1)

def load_and_validate():
    
    print("Loading data...")
    try:
        voice = pd.read_csv("softmax_voice.csv")
        phys = pd.read_csv("softmax_wesad.csv")
        text = pd.read_csv("softmax_dass.csv")
        
        
        print("Voice columns:", voice.columns.tolist())
        print("Phys columns:", phys.columns.tolist())
        print("Text columns:", text.columns.tolist())
        
        
        return voice, phys, text
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise

def preprocess(voice, phys, text):
    
    print("Preprocessing...")
    
    rename_map = {
        'prob_Low':'low_prob',
        'prob_Medium':'medium_prob',
        'prob_High':'high_prob',
        'Low':'low_prob',
        'Medium':'medium_prob',
        'High':'high_prob',
        'low':'low_prob',
        'medium':'medium_prob',
        'high':'high_prob',
        'Low_prob':'low_prob',
        'Medium_prob':'medium_prob',
        'High_prob':'high_prob',
        'label':'true_label',
        'Label':'true_label',
        'target':'true_label',
        'Target':'true_label'
    }
    for df in [voice, phys, text]:
        df.rename(columns=rename_map, inplace=True)
        if 'SampleID' not in df.columns:
            df.insert(0, 'SampleID', range(len(df)))
        df['true_label'] = df['true_label'].astype(int)  # Ensure int labels
        df.set_index('SampleID', inplace=True)
    
    
    req_cols = {'low_prob','medium_prob','high_prob','true_label'}
    for df, name in zip([voice,phys,text], ['voice','phys','text']):
        if not req_cols.issubset(df.columns):
            print(f"Missing columns in {name} data. Available: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns in {name} data")
    
    return voice, phys, text

def augment_dataset(df, label_col='true_label'):
    aug_samples = []
    class_counts = df[label_col].value_counts()
    
    
    aug_params = {
        0: 100,  
        1: 300,  
        2: 100   
    }
    
    for class_id, n in aug_params.items():
        if class_id not in class_counts:
            continue
            
        df_class = df[df[label_col] == class_id]
        if len(df_class) < 2:
            continue
            
        for _ in range(n):
            
            sampled = resample(df_class, n_samples=2, replace=True)
            interp = (
                sampled.iloc[0][['low_prob','medium_prob','high_prob']].values + 
                sampled.iloc[1][['low_prob','medium_prob','high_prob']].values
            ) / 2
            interp += np.random.normal(0, 0.02, size=3)
            interp = np.clip(interp, 0, 1)
            interp /= interp.sum()
            
            aug_samples.append({
                'low_prob': interp[0],
                'medium_prob': interp[1],
                'high_prob': interp[2],
                'true_label': class_id
            })
    
    return pd.concat([df, pd.DataFrame(aug_samples)], ignore_index=True)

def calculate_weights(labels):
    
    counts = np.bincount(labels, minlength=3)
    return (1 / (counts + 1e-6)) / sum(1 / (counts + 1e-6))  # Smooth division

def perform_fusion(voice, phys, text, class_weights):
    
    print("Fusing with physiological bias...")
    results = []
    feature_vectors = []  # For SHAP analysis
    all_ids = set(voice.index) | set(phys.index) | set(text.index)
    fusion = PhysioDominantFusion(class_weights=dict(enumerate(class_weights)))
    
    for sid in sorted(all_ids):
        mod_probs = {}
        label = None
        
        
        if sid in phys.index:
            mod_probs['phys'] = phys.loc[sid][['low_prob','medium_prob','high_prob']].values
            label = int(phys.loc[sid]['true_label'])
        
        
        if sid in text.index:
            mod_probs['text'] = text.loc[sid][['low_prob','medium_prob','high_prob']].values
            if label is None:
                label = int(text.loc[sid]['true_label'])
        
        
        if sid in voice.index:
            mod_probs['voice'] = voice.loc[sid][['low_prob','medium_prob','high_prob']].values
            if label is None:
                label = int(voice.loc[sid]['true_label'])
        
        if mod_probs:
            pred = fusion.predict(mod_probs)
            proba = fusion.predict_proba(mod_probs)
            feature_vec = fusion._convert_to_feature_vector(mod_probs)
            
            results.append((sid, *proba, pred, label))
            feature_vectors.append(feature_vec)
    
    df = pd.DataFrame(results, columns=['SampleID','low_prob','medium_prob','high_prob','Prediction','True_Label'])
    feature_matrix = np.array(feature_vectors)
    
    return df, feature_matrix, fusion

def generate_learning_curves(feature_matrix, labels, fusion_model, output_dir):
    
    print("Generating learning curves...")
    
    
    fusion_model.fit(feature_matrix, labels)
    
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        fusion_model, 
        feature_matrix, 
        labels,
        train_sizes=train_sizes,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='red', label='Training Score')
    plt.fill_between(train_sizes_abs, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color='red')
    
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='green', label='Cross-validation Score')
    plt.fill_between(train_sizes_abs, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Balanced Accuracy Score')
    plt.title('Learning Curves (Physio-Dominant Fusion)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
    learning_data = pd.DataFrame({
        'train_size': train_sizes_abs,
        'train_score_mean': train_scores_mean,
        'train_score_std': train_scores_std,
        'test_score_mean': test_scores_mean,
        'test_score_std': test_scores_std
    })
    learning_data.to_csv(f"{output_dir}/learning_curves_data.csv", index=False)
    
    return train_scores_mean, test_scores_mean

def generate_validation_curves(feature_matrix, labels, fusion_model, output_dir):
    print("Generating validation curves...")
    phys_weights = np.linspace(0.3, 0.8, 10)
    
    train_scores_list = []
    test_scores_list = []
    
    for phys_weight in phys_weights:
        
        temp_fusion = PhysioDominantFusion(class_weights=fusion_model.class_weights)
        temp_fusion.mod_weights = {
            'phys': phys_weight,
            'text': (1 - phys_weight) * 0.6,
            'voice': (1 - phys_weight) * 0.4
        }
        
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            temp_fusion, 
            feature_matrix, 
            labels,
            train_sizes=[0.8],  # Use 80% of data
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        
        train_scores_list.append(np.mean(train_scores))
        test_scores_list.append(np.mean(test_scores))
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(phys_weights, train_scores_list, 'o-', color='red', label='Training Score')
    plt.plot(phys_weights, test_scores_list, 'o-', color='green', label='Validation Score')
    
    plt.xlabel('Physiological Weight')
    plt.ylabel('Balanced Accuracy Score')
    plt.title('Validation Curves - Physiological Weight Impact')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/validation_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
    validation_data = pd.DataFrame({
        'phys_weight': phys_weights,
        'train_score': train_scores_list,
        'test_score': test_scores_list
    })
    validation_data.to_csv(f"{output_dir}/validation_curves_data.csv", index=False)
    
    return train_scores_list, test_scores_list

def generate_shap_analysis(feature_matrix, fusion_model, output_dir):
    
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping SHAP analysis")
        return
    
    print("Generating SHAP analysis...")
    try:
        
        def model_wrapper(X):
            
            probas = fusion_model.predict_proba_from_features(X)
            return probas[:, 1]  
        
        
        sample_size = min(100, len(feature_matrix))
        sample_indices = np.random.choice(len(feature_matrix), sample_size, replace=False)
        sample_features = feature_matrix[sample_indices]
        
        
        background = shap.sample(sample_features, 20)  
        explainer = shap.Explainer(model_wrapper, background)
        
        
        shap_values = explainer(sample_features[:50])  
        
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_features[:50], 
                         feature_names=fusion_model.feature_names,
                         show=False)
        plt.title("SHAP Feature Importance (Medium Stress)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample_features[:50],
                         feature_names=fusion_model.feature_names,
                         plot_type="bar", show=False)
        plt.title("SHAP Feature Importance Bar Plot")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        plt.title("SHAP Waterfall Plot - Sample 1")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_waterfall.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        
        shap_df = pd.DataFrame(
            shap_values.values,
            columns=fusion_model.feature_names
        )
        shap_df.to_csv(f"{output_dir}/shap_values.csv", index=False)
        
        
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        importance_df = pd.DataFrame({
            'Feature': fusion_model.feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        
        
        print("Analyzing individual classes...")
        for class_idx, class_name in enumerate(['Low', 'Medium', 'High']):
            def class_wrapper(X):
                probas = fusion_model.predict_proba_from_features(X)
                return probas[:, class_idx]
            
            class_explainer = shap.Explainer(class_wrapper, background)
            class_shap_values = class_explainer(sample_features[:20])
            
            
            class_df = pd.DataFrame(
                class_shap_values.values,
                columns=fusion_model.feature_names
            )
            class_df.to_csv(f"{output_dir}/shap_values_{class_name.lower()}.csv", index=False)
        
        print("SHAP analysis completed successfully")
        
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

def evaluate_and_save(df, feature_matrix, fusion_model, output_dir):
    """Generate all outputs including balanced accuracy and learning curves"""
    print("Evaluating results...")
    os.makedirs(output_dir, exist_ok=True)
    df.set_index('SampleID', inplace=True)
    df.to_csv(f"{output_dir}/predictions.csv")
    
    balanced_acc = balanced_accuracy_score(df['True_Label'], df['Prediction'])
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    report = classification_report(
        df['True_Label'], 
        df['Prediction'],
        target_names=['Low','Medium','High'],
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/classification_report.csv")
    metrics_df = pd.DataFrame({
        'Metric': ['Balanced_Accuracy', 'Accuracy', 'Macro_F1', 'Weighted_F1'],
        'Value': [
            balanced_acc,
            report['accuracy'],
            report['macro avg']['f1-score'],
            report['weighted avg']['f1-score']
        ]
    })
    metrics_df.to_csv(f"{output_dir}/balanced_accuracy.csv", index=False)
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(df['True_Label'], df['Prediction'])
    sns.heatmap(
        cm,
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Low','Medium','High'],
        yticklabels=['Low','Medium','High']
    )
    plt.title(f"Confusion Matrix (Physio-Dominant)\nBalanced Accuracy: {balanced_acc:.4f}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
    y_true = label_binarize(df['True_Label'], classes=[0,1,2])
    y_score = df[['low_prob','medium_prob','high_prob']].values
    
    plt.figure(figsize=(8,6))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true[:,i], y_score[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{['Low','Medium','High'][i]} (AUC={roc_auc:.2f})")
    
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    
    train_scores, test_scores = generate_learning_curves(
        feature_matrix, df['True_Label'].values, fusion_model, output_dir
    )
    
    
    train_val_scores, test_val_scores = generate_validation_curves(
        feature_matrix, df['True_Label'].values, fusion_model, output_dir
    )
    
    
    generate_shap_analysis(feature_matrix, fusion_model, output_dir)
    
    
    with open(f"{output_dir}/fusion_model.pkl", "wb") as f:
        pickle.dump(fusion_model, f)
    
    
    summary = {
        'balanced_accuracy': balanced_acc,
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'final_train_score': train_scores[-1],
        'final_test_score': test_scores[-1],
        'best_validation_score': max(test_val_scores),
        'total_samples': len(df),
        'class_distribution': df['True_Label'].value_counts().to_dict()
    }
    
    with open(f"{output_dir}/results_summary.txt", "w") as f:
        f.write("=== SafeSpace Fusion Results Summary ===\n")
        f.write(f"Balanced Accuracy: {summary['balanced_accuracy']:.4f}\n")
        f.write(f"Standard Accuracy: {summary['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {summary['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {summary['weighted_f1']:.4f}\n")
        f.write(f"Final Training Score: {summary['final_train_score']:.4f}\n")
        f.write(f"Final Test Score: {summary['final_test_score']:.4f}\n")
        f.write(f"Best Validation Score: {summary['best_validation_score']:.4f}\n")
        f.write(f"Total Samples: {summary['total_samples']}\n")
        f.write(f"Class Distribution: {summary['class_distribution']}\n")
    
    
    with ZipFile("physio_dominant_results.zip", "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), output_dir)
                )

def main():
    try:
        
        voice, phys, text = load_and_validate()
        voice, phys, text = preprocess(voice, phys, text)
        
        
        print("Augmenting data...")
        voice = augment_dataset(voice)
        phys = augment_dataset(phys)
        text = augment_dataset(text)
        
        
        all_labels = pd.concat([voice['true_label'], phys['true_label'], text['true_label']])
        class_weights = calculate_weights(all_labels)
        
        
        results, feature_matrix, fusion_model = perform_fusion(voice, phys, text, class_weights)
        
        
        evaluate_and_save(results, feature_matrix, fusion_model, "physio_dominant_results")
        
        print("\n✅ Physiology-dominant fusion completed!")
        print(f"Results saved to: physio_dominant_results/")
        print(f"Zip archive created: physio_dominant_results.zip")
        print("\nNew outputs added:")
        print("- balanced_accuracy.csv: Balanced accuracy and other metrics")
        print("- learning_curves.png: Training vs validation curves")
        print("- validation_curves.png: Hyperparameter validation curves")
        print("- learning_curves_data.csv: Raw learning curve data")
        print("- validation_curves_data.csv: Raw validation curve data")
        print("- results_summary.txt: Comprehensive results summary")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        import sklearn
        if sklearn.__version__ != '1.3.0':
            print(f"Warning: Using scikit-learn {sklearn.__version__}, tested with 1.3.0")
    except ImportError:
        print("Error: scikit-learn not installed")
        exit(1)
    
    main()