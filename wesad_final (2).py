
import os
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, validation_curve
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import joblib
import warnings
from itertools import cycle
warnings.filterwarnings('ignore')
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
import json

CFG = {
    "root": "WESAD",
    "orig_fs": 700,
    "fs": 100,
    "window_sec": 10,
    "stride_sec": 5,
    "sensors": ["ECG", "EDA", "EMG", "Temp"], 
    "label_map": {0: 0, 3: 0, 2: 1, 1: 2},  
    "class_names": ["Low", "Medium", "High"],
    "seed": 42,
    "test_size": 0.2,  
    "validation_size": 0.2, 
    "max_train_val_gap": 0.05,
    "early_stopping_patience": 10,
    "min_samples_for_complexity": 1000,
}

DOWN_F = CFG["orig_fs"] // CFG["fs"]
STEP = CFG["window_sec"] * CFG["fs"]
STRIDE = CFG["stride_sec"] * CFG["fs"]
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])

def clean_data(X):
    try:
        X = np.asarray(X)
        if X.dtype.kind not in 'f':
            X = X.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=np.nanmax(X[X != np.inf]), neginf=np.nanmin(X[X != -np.inf]))
        return X
    except Exception as e:
        print(f"⚠ Error in clean_data: {str(e)}")
        return np.zeros_like(X) if isinstance(X, np.ndarray) else np.zeros((1, 180))

def is_valid_data(X):
    try:
        X = np.asarray(X)
        if X.dtype.kind not in 'biufc':
            return False
        return np.isfinite(X).all()
    except:
        return False

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def downsample(x):
    return x[::DOWN_F]

def load_subject(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        chest_data = data.get('signal', {}).get('chest', {})
        labels = data.get('label', np.array([])).astype(int)

        if len(labels) == 0:
            raise ValueError("No labels found")

        labels = downsample(labels)

        signals = {}
        for sensor in CFG["sensors"]:
            if sensor not in chest_data:
                print(f"   ⚠ Missing sensor {sensor}")
                continue
            sig = chest_data[sensor]
            signals[sensor] = zscore(sig.flatten())

        return signals, labels

    except Exception as e:
        print(f"⚠ Failed to load {path}: {str(e)}")
        return {}, np.array([])
    
def extract_time_features(signal_data):
    try:
        signal_data = np.asarray(signal_data).astype(float)
        features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.var(signal_data),
            skew(signal_data),
            kurtosis(signal_data),
            np.min(signal_data),
            np.max(signal_data),
            np.ptp(signal_data),
            np.median(signal_data),
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75),
            np.mean(np.abs(np.diff(signal_data))),
            np.sqrt(np.mean(signal_data**2))
        ]
        return features
    except:
        return [0.0] * 13

def extract_freq_features(signal_data, fs=100):
    try:
        signal_data = np.asarray(signal_data).astype(float)
        if len(signal_data) < 8:
            return [0.0] * 11
        
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)//4))
        
        bands = {
            'very_low': (0.0, 0.04),
            'low': (0.04, 0.15),
            'mid': (0.15, 0.4),
            'high': (0.4, 0.5)
        }
        
        total_power = np.sum(psd)
        features = []
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            features.append(band_power)
            features.append(band_power / (total_power + 1e-8))
        
        features.extend([
            np.mean(freqs),
            np.std(freqs),
            freqs[np.argmax(psd)]
        ])
        return features
    except:
        return [0.0] * 11

def extract_wavelet_features(signal_data):
    try:
        signal_data = np.asarray(signal_data).astype(float)
        coeffs = pywt.wavedec(signal_data, 'db4', level=4)
        features = []
        for coeff in coeffs:
            if len(coeff) > 0:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.var(coeff),
                    np.max(np.abs(coeff))
                ])
        return features if features else [0.0] * 16
    except:
        return [0.0] * 16

def extract_ecg_features(signal_data, fs=100):
    try:
        signal_data = np.asarray(signal_data).astype(float)
        peaks, _ = signal.find_peaks(signal_data, height=np.std(signal_data), distance=fs//3)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs * 1000
            return [
                np.mean(rr_intervals),
                np.std(rr_intervals),
                np.sqrt(np.mean(np.diff(rr_intervals)**2)),
                len(peaks) / (len(signal_data) / fs) * 60
            ]
        return [0.0] * 4
    except:
        return [0.0] * 4

def extract_window_features(sigs):
    features = []
    feature_names = []
    
    for sensor, signal_data in sigs.items():
        time_feats = extract_time_features(signal_data)
        features.extend(time_feats)
        time_names = [f"{sensor}_mean", f"{sensor}_std", f"{sensor}_var", f"{sensor}_skew", 
                     f"{sensor}_kurt", f"{sensor}_min", f"{sensor}_max", f"{sensor}_ptp",
                     f"{sensor}_median", f"{sensor}_q25", f"{sensor}_q75", f"{sensor}_mad", f"{sensor}_rms"]
        feature_names.extend(time_names)
        freq_feats = extract_freq_features(signal_data)
        features.extend(freq_feats)
        freq_names = [f"{sensor}_vl_power", f"{sensor}_vl_rel", f"{sensor}_l_power", f"{sensor}_l_rel",
                     f"{sensor}_m_power", f"{sensor}_m_rel", f"{sensor}_h_power", f"{sensor}_h_rel",
                     f"{sensor}_freq_mean", f"{sensor}_freq_std", f"{sensor}_dom_freq"]
        feature_names.extend(freq_names)
        wavelet_feats = extract_wavelet_features(signal_data)
        features.extend(wavelet_feats)
        wavelet_names = [f"{sensor}wl{i}" for i in range(len(wavelet_feats))]
        feature_names.extend(wavelet_names)
        
        if sensor == "ECG":
            ecg_feats = extract_ecg_features(signal_data)
            features.extend(ecg_feats)
            ecg_names = [f"{sensor}_mean_rr", f"{sensor}_sdnn", f"{sensor}_rmssd", f"{sensor}_hr"]
            feature_names.extend(ecg_names)
    
    features = np.array(features, dtype=float)
    return clean_data(features), feature_names

def slice_windows_with_features(sigs, labels, subject_id):
    n_win = (len(labels) - STEP) // STRIDE + 1
    X, y, subjects = [], [], []
    
    for i in range(n_win):
        i0, i1 = i * STRIDE, i * STRIDE + STEP
        label = np.bincount(labels[i0:i1]).argmax()
        if label not in CFG["label_map"]:
            continue
        
        window_sigs = {ch: np.asarray(sigs[ch][i0:i1]).astype(float) for ch in sigs}
        features, _ = extract_window_features(window_sigs)
        
        X.append(features)
        y.append(CFG["label_map"][label])
        subjects.append(subject_id)
    
    X = np.array(X, dtype=float)
    y = np.array(y)
    subjects = np.array(subjects)
    return clean_data(X), y, subjects

def build_global_dataset():
    paths = sorted(Path(CFG["root"]).rglob("S*.pkl"))
    print(f"\n🧪 Found {len(paths)} subjects.")
    
    all_X, all_y, all_subjects = [], [], []
    
    for path in paths:
        try:
            sigs, labs = load_subject(path)
            subject_id = path.stem
            print(f"➡ {subject_id}: Labels={np.unique(labs)}")
            
            if not sigs: 
                print(f"   ⚠ No valid signals found, skipping...")
                continue
                
            X, y, subjects = slice_windows_with_features(sigs, labs, subject_id)
            print(f"   - Windows extracted: {len(X)}, Features: {X.shape[1] if len(X) > 0 else 0}")
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_subjects.append(subjects)
        except Exception as e:
            print(f"❌ Error processing {path.stem}: {str(e)}")
            continue
    
    if not all_X:
        raise ValueError("No valid data found!")
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    subjects = np.concatenate(all_subjects)
    
    print(f"\n📊 Global dataset created:")
    print(f"   Total samples: {len(X)}")
    print(f"   Total features: {X.shape[1]}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   Subjects: {len(np.unique(subjects))}")
    
    return X, y, subjects

def create_regularized_models(n_samples, n_features):
    if n_samples < CFG["min_samples_for_complexity"]:
        max_features_rf = min(20, n_features // 4)
        max_features_gb = min(15, n_features // 5)
        max_features_svm = min(10, n_features // 6)
    else:
        max_features_rf = min(50, n_features // 3)
        max_features_gb = min(30, n_features // 4)
        max_features_svm = min(20, n_features // 5)
    
    print(f"🔧 Adaptive feature selection: RF={max_features_rf}, GB={max_features_gb}, SVM={max_features_svm}")
    
    models = {
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=max_features_rf)),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                max_samples=0.8,
                random_state=CFG["seed"],
                class_weight='balanced'
            ))
        ]),
        
        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=max_features_gb)),
            ('classifier', GradientBoostingClassifier(
                n_estimators=30,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.7,
                max_features='sqrt',
                random_state=CFG["seed"],
                n_iter_no_change=CFG["early_stopping_patience"],
                validation_fraction=0.15,
                tol=1e-4
            ))
        ]),
        
        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=max_features_svm)),
            ('classifier', SVC(
                kernel='rbf',
                C=0.5,
                gamma='scale',
                probability=True,
                random_state=CFG["seed"],
                class_weight='balanced'
            ))
        ]),
        
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=min(25, n_features // 4))),
            ('classifier', LogisticRegression(
                C=0.5,
                penalty='l2',
                max_iter=1000,
                random_state=CFG["seed"],
                class_weight='balanced'
            ))
        ]),
        
        'extra_trees': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=max_features_rf)),
            ('classifier', RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='log2',
                bootstrap=False,
                random_state=CFG["seed"],
                class_weight='balanced'
            ))
        ])
    }
    
    return models

def enhanced_overfitting_detection(model, X_train, y_train, X_val, y_val, model_name):
    print(f"🔍 Enhanced overfitting check for {model_name}...")

    try:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_acc = balanced_accuracy_score(y_train, train_pred)
        val_acc = balanced_accuracy_score(y_val, val_pred)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=StratifiedKFold(n_splits=7, shuffle=True, random_state=CFG["seed"]),
                                    scoring='balanced_accuracy')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        overfitting_gap = train_acc - val_acc
        high_variance = cv_std > 0.08
        perfect_train = train_acc >= 0.999

        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Validation Accuracy: {val_acc:.3f}")
        print(f"   CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"   Overfitting Gap: {overfitting_gap:.3f}")

        overfitting_risk = "LOW"
        penalty = 0.0

        if perfect_train:
            print(f"   🚨 PERFECT TRAINING ACCURACY: Major overfitting risk!")
            overfitting_risk = "SEVERE"
            penalty += 0.15
        elif overfitting_gap > CFG["max_train_val_gap"]:
            if overfitting_gap > 0.10:
                print(f"   ⚠  HIGH OVERFITTING RISK: Gap > 0.10")
                overfitting_risk = "HIGH"
                penalty += 0.10
            else:
                print(f"   ⚠  MODERATE OVERFITTING RISK: Gap > {CFG['max_train_val_gap']}")
                overfitting_risk = "MODERATE"
                penalty += 0.05
        else:
            print(f"   ✅ Low overfitting risk")

        if high_variance:
            print(f"   ⚠  HIGH VARIANCE: CV std > 0.08")
            penalty += 0.03
        else:
            print(f"   ✅ Low variance across folds")

        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'overfitting_gap': overfitting_gap,
            'overfitting_risk': overfitting_risk,
            'perfect_train': perfect_train,
            'high_variance': high_variance,
            'penalty': penalty,
            'adjusted_score': val_acc - penalty
        }

    except Exception as e:
        print(f"   ❌ Overfitting detection failed: {str(e)}")
        return None

def select_best_regularized_model(X_train, y_train, X_val, y_val):
    n_samples, n_features = X_train.shape
    models = create_regularized_models(n_samples, n_features)
    try:
        X_train = np.asarray(X_train, dtype=float)
        X_val = np.asarray(X_val, dtype=float)
        
        if not is_valid_data(X_train) or not is_valid_data(X_val):
            print("⚠  Invalid data detected - applying aggressive cleaning")
            X_train = clean_data(X_train)
            X_val = clean_data(X_val)
        
        print(f"📊 Training set size: {n_samples} samples, {n_features} features")
        print(f"📊 Validation set size: {len(X_val)} samples")
        if n_samples < CFG["min_samples_for_complexity"]:
            print("⚠  SMALL DATASET: Using conservative model parameters")
        if n_features > n_samples // 2:
            print("⚠  HIGH DIMENSIONALITY: Using aggressive feature selection")
            
    except Exception as e:
        print(f"❌ Data validation failed: {str(e)}")
        return None, "invalid_data", None
    
    best_model = None
    best_score = -np.inf
    best_name = ""
    best_stats = None
    
    print("\n🔍 Evaluating regularized models...")
    
    for name, model in models.items():
        try:
            stats = enhanced_overfitting_detection(model, X_train, y_train, X_val, y_val, name)
            if stats is None:
                continue
            
            adjusted_score = stats['adjusted_score']
            print(f"   Adjusted Score: {adjusted_score:.3f} (penalty: {stats['penalty']:.3f})")
            if stats['overfitting_risk'] == "SEVERE":
                print(f"   ❌ REJECTED: Severe overfitting risk")
                continue
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_model = model
                best_name = name
                best_stats = stats
            
            print()
            
        except Exception as e:
            print(f"   ❌ {name}: Failed - {str(e)}")
            continue
    
    if best_model is None:
        print("🚨 All models failed! Using conservative fallback...")
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=min(15, n_features // 6))),
            ('classifier', LogisticRegression(
                C=0.1,
                penalty='l2',
                max_iter=1000,
                random_state=CFG["seed"],
                class_weight='balanced'
            ))
        ])
        best_name = "conservative_logistic_regression"
        
        try:
            stats = enhanced_overfitting_detection(best_model, X_train, y_train, X_val, y_val, best_name)
            best_stats = stats
        except Exception as e:
            print(f"❌ Fallback model failed: {str(e)}")
            return None, "failed_fallback", None
    
    return best_model, best_name, best_stats

def plot_learning_curves(model, X_train, y_train, model_name):
    try:
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='balanced_accuracy', random_state=CFG["seed"]
        )
        
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Balanced Accuracy')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/plots/learning_curves_{model_name}.png")
        plt.close()
        
        print(f"📈 Learning curves saved for {model_name}")
        
    except Exception as e:
        print(f"⚠ Could not plot learning curves: {str(e)}")

print("🚀 Starting Regularized Single Global Model Training...")

X, y, subjects = build_global_dataset()

X_temp, X_test, y_temp, y_test, subjects_temp, subjects_test = train_test_split(
    X, y, subjects, test_size=CFG["test_size"], 
    stratify=y, random_state=CFG["seed"]
)

X_train, X_val, y_train, y_val, subjects_train, subjects_val = train_test_split(
    X_temp, y_temp, subjects_temp, test_size=CFG["validation_size"]/(1-CFG["test_size"]),
    stratify=y_temp, random_state=CFG["seed"]
)

print(f"\n📊 Data split completed:")
print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")
print(f"   Train subjects: {len(np.unique(subjects_train))}")
print(f"   Val subjects: {len(np.unique(subjects_val))}")
print(f"   Test subjects: {len(np.unique(subjects_test))}")

print(f"\n=== Training Regularized Global Model ===")
best_model, model_name, model_stats = select_best_regularized_model(X_train, y_train, X_val, y_val)

if best_model is None:
    print("❌ No valid model could be trained!")
    exit()

print(f"\n🎯 Selected Model: {model_name}")
print(f"🎯 Overfitting Risk: {model_stats['overfitting_risk']}")

plot_learning_curves(best_model, X_train, y_train, model_name)
print("\n📈 Retraining on combined train+validation data...")
X_train_final = np.concatenate([X_train, X_val])
y_train_final = np.concatenate([y_train, y_val])

best_model.fit(X_train_final, y_train_final)
joblib.dump(best_model, "results/models/regularized_global_model.pkl")
print("✅ Regularized global model saved!")
RESULT_DIR = "result_WESAD"
os.makedirs(RESULT_DIR, exist_ok=True)

def interpret_model(model, X_train, X_test, y_test, feature_names, model_name):
    print("\n🔍 Applying XAI techniques for model interpretation...")
    
    try:
        
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        
        if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:
            selector = model.named_steps['feature_selection']
            selected_mask = selector.get_support()
            selected_features = np.array(feature_names)[selected_mask]
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
        else:
            selected_features = feature_names
            X_train_selected = X_train
            X_test_selected = X_test
        
        
        if hasattr(classifier, 'feature_importances_'):
            print("\n📊 Feature Importance Analysis:")
            importances = classifier.feature_importances_
            std = np.std([tree.feature_importances_ for tree in classifier.estimators_]
                        if hasattr(classifier, 'estimators_') else [importances], axis=0)
            
            
            fi_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances,
                'std': std
            }).sort_values('importance', ascending=False)
            
            
            fi_df.to_csv(os.path.join(RESULT_DIR, "feature_importances.csv"), index=False)
            print("✅ Feature importances saved to CSV")
            
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', 
                        data=fi_df.head(20), palette='viridis')
            plt.title(f"Top 20 Important Features - {model_name}")
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "feature_importances.png"))
            plt.close()
            print("✅ Feature importance plot saved")
        
        
        try:
            import shap
            print("\n📊 Calculating SHAP values...")
            
            
            if hasattr(classifier, 'feature_importances_'):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_train_selected)
                
            
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_train_selected, 
                                 feature_names=selected_features,
                                 class_names=CFG['class_names'],
                                 plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, "shap_feature_importance.png"))
                plt.close()
                
                
                for i, class_name in enumerate(CFG['class_names']):
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values[i], X_train_selected, 
                                    feature_names=selected_features,
                                    show=False)
                    plt.title(f"SHAP Values - {class_name}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULT_DIR, f"shap_values_{class_name}.png"))
                    plt.close()
                
                print("✅ SHAP summary plots saved")
                
                
                sample_idx = np.random.choice(len(X_test_selected), size=5, replace=False)
                for idx in sample_idx:
                    plt.figure()
                    shap.decision_plot(explainer.expected_value[i], 
                                     shap_values[i][idx:idx+1], 
                                     selected_features,
                                     ignore_warnings=True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULT_DIR, f"shap_decision_plot_sample_{idx}.png"))
                    plt.close()
                
            
            elif hasattr(classifier, 'coef_'):
                explainer = shap.LinearExplainer(classifier, X_train_selected)
                shap_values = explainer.shap_values(X_train_selected)
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_train_selected, 
                                 feature_names=selected_features,
                                 show=False)
                plt.title("SHAP Values (Linear Model)")
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, "shap_values_linear.png"))
                plt.close()
                
            print("✅ SHAP analysis completed")
            
        except ImportError:
            print("⚠ SHAP not installed. Install with: pip install shap")
        except Exception as e:
            print(f"⚠ SHAP analysis failed: {str(e)}")
        
        
        try:
            from sklearn.inspection import PartialDependenceDisplay
            print("\n📊 Generating Partial Dependence Plots...")
            
            if hasattr(classifier, 'feature_importances_'):
                top_features = fi_df.head(3)['feature'].tolist()
                feature_indices = [np.where(selected_features == f)[0][0] for f in top_features]
                
                for i, feat_idx in enumerate(feature_indices):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    PartialDependenceDisplay.from_estimator(
                        model, X_train, 
                        features=[feat_idx],
                        feature_names=feature_names,
                        target=0,  # First class
                        ax=ax
                    )
                    plt.title(f"Partial Dependence - {top_features[i]}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULT_DIR, f"partial_dependence_{top_features[i]}.png"))
                    plt.close()
                    
                print("✅ Partial dependence plots saved")
                
        except Exception as e:
            print(f"⚠ Partial dependence plots failed: {str(e)}")
        
        
        try:
            from sklearn.inspection import permutation_importance
            print("\n📊 Calculating Permutation Importance...")
            
            result = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=CFG["seed"],
                n_jobs=-1
            )
            
            perm_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            perm_imp_df.to_csv(os.path.join(RESULT_DIR, "permutation_importance.csv"), index=False)
            
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance_mean', y='feature', 
                        data=perm_imp_df.head(20), palette='rocket')
            plt.title("Top 20 Features by Permutation Importance")
            plt.xlabel("Mean Decrease in Accuracy")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "permutation_importance.png"))
            plt.close()
            
            print("✅ Permutation importance analysis saved")
            
        except Exception as e:
            print(f"⚠ Permutation importance failed: {str(e)}")
        
        
        try:
            import lime
            import lime.lime_tabular
            print("\n📊 Generating LIME explanations...")
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_selected,
                feature_names=selected_features,
                class_names=CFG['class_names'],
                mode='classification',
                random_state=CFG["seed"]
            )
            
           
            sample_idx = np.random.choice(len(X_test_selected), size=3, replace=False)
            for i, idx in enumerate(sample_idx):
                exp = explainer.explain_instance(
                    X_test_selected[idx], 
                    model.predict_proba,
                    num_features=10,
                    top_labels=1
                )
                
               
                html = exp.as_html()
                with open(os.path.join(RESULT_DIR, f"lime_explanation_sample_{i}.html"), 'w') as f:
                    f.write(html)
                
                
                fig = exp.as_pyplot_figure()
                fig.tight_layout()
                fig.savefig(os.path.join(RESULT_DIR, f"lime_explanation_sample_{i}.png"))
                plt.close()
            
            print("✅ LIME explanations saved for 3 samples")
            
        except ImportError:
            print("⚠ LIME not installed. Install with: pip install lime")
        except Exception as e:
            print(f"⚠ LIME failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ XAI interpretation failed: {str(e)}")


if 'feature_names' not in locals():
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
interpret_model(best_model, X_train, X_test, y_test, feature_names, model_name)


print("\n🧪 Evaluating on test set...")

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

test_acc = balanced_accuracy_score(y_test, y_pred)
final_cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 FINAL RESULTS:")
print(f"Final Test Accuracy: {test_acc:.3f}")
print(f"Model: {model_name}")
print(f"Overfitting Risk: {model_stats['overfitting_risk']}")

if model_stats:
    print(f"\n🔍 MODEL PERFORMANCE ANALYSIS:")
    print(f"   Training Accuracy: {model_stats['train_acc']:.3f}")
    print(f"   Validation Accuracy: {model_stats['val_acc']:.3f}")
    print(f"   Test Accuracy: {test_acc:.3f}")
    print(f"   Overfitting Gap (Train-Val): {model_stats['overfitting_gap']:.3f}")
    print(f"   Generalization Gap (Val-Test): {model_stats['val_acc'] - test_acc:.3f}")
    if model_stats['overfitting_gap'] <= CFG["max_train_val_gap"]:
        print("   ✅ OVERFITTING: Well controlled")
    else:
        print("   ⚠  OVERFITTING: Still present but reduced")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CFG['class_names']))
RESULT_DIR = "result_WESAD"
os.makedirs(RESULT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(RESULT_DIR, "regularized_global_model.pkl"))
print("✅ Model saved to result_WESAD!")
proba_df = pd.DataFrame(y_proba, columns=[f"prob_{cls}" for cls in CFG['class_names']])
proba_df["true_label"] = y_test
proba_df["predicted_label"] = y_pred
proba_df.to_csv(os.path.join(RESULT_DIR, "softmax_outputs.csv"), index=False)
print("✅ Softmax outputs saved.")
report_dict = classification_report(y_test, y_pred, target_names=CFG['class_names'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RESULT_DIR, "classification_report.csv"))
print("✅ Classification report saved.")
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CFG['class_names'], yticklabels=CFG['class_names'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()
print("✅ Confusion matrix plot saved.")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, roc_auc = dict(), dict(), dict()
colors = cycle(['blue', 'green', 'red'])

plt.figure(figsize=(8, 6))
for i, color in zip(range(len(CFG['class_names'])), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{CFG['class_names'][i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"))
plt.close()
print("✅ ROC curve plot saved.")
metrics_summary = {
    "test_balanced_accuracy": float(test_acc),
    "train_accuracy": float(model_stats['train_acc']),
    "val_accuracy": float(model_stats['val_acc']),
    "cv_mean": float(model_stats['cv_mean']),
    "cv_std": float(model_stats['cv_std']),
    "overfitting_gap": float(model_stats['overfitting_gap']),
    "val_minus_test_gap": float(model_stats['val_acc'] - test_acc),
    "overfitting_risk": model_stats['overfitting_risk']
}
with open(os.path.join(RESULT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=4)
print("✅ All evaluation metrics saved.")
print("\n📦 Generating softmax outputs for ALL samples (for late fusion)...")
best_model.fit(X, y)
y_all_proba = best_model.predict_proba(X)
y_all_pred = best_model.predict(X)
all_proba_df = pd.DataFrame(y_all_proba, columns=[f"prob_{cls}" for cls in CFG['class_names']])
all_proba_df["true_label"] = y
all_proba_df["predicted_label"] = y_all_pred
all_proba_df.to_csv(os.path.join(RESULT_DIR, "softmax_outputs_ALL.csv"), index=False)

print("✅ Softmax outputs for all samples saved to 'softmax_outputs_ALL.csv'")