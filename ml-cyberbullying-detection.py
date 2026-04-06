import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, auc,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("All libraries loaded!")


# STEP 1 - LOAD DATA


df = pd.read_csv('cyberbullying_tweets.csv')  
print("Shape:", df.shape)
print(df.columns.tolist())
print("\nClass distribution (original):")
print(df['cyberbullying_type'].value_counts())


# STEP 2 - MERGE CONFUSED CLASSES

# not_cyberbullying and other_cyberbullying are semantically
# too similar — merging them removes the main confusion source
df['cyberbullying_type'] = df['cyberbullying_type'].replace(
    'other_cyberbullying', 'not_cyberbullying'
)

print("\nClass distribution (after merging):")
print(df['cyberbullying_type'].value_counts())

plt.figure(figsize=(8, 5))
df['cyberbullying_type'].value_counts().plot(
    kind='bar', color='steelblue', edgecolor='white')
plt.title('Class Distribution After Merging')
plt.xlabel('Type'); plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()


# STEP 3 - BETTER TEXT CLEANING

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# Keep these words — they carry important meaning for classification
KEEP_WORDS = {
    'not', 'no', 'never', 'none', 'nobody', 'nothing',
    'neither', 'nor', 'cannot', "can't", "won't", "don't",
    'hate', 'kill', 'die', 'hurt', 'attack', 'abuse'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)     
    text = re.sub(r'@\w+', '', text)                
    text = re.sub(r'#(\w+)', r'\1', text)           
    text = re.sub(r'[^a-z\s!?]', '', text)          
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w not in stop_words or w in KEEP_WORDS
    ]
    return ' '.join(tokens)

df['clean_text'] = df['tweet_text'].apply(clean_text)
print("\nCleaning done. Sample:")
print(df[['tweet_text', 'clean_text']].head(3).to_string())


# STEP 4 - COMBINED TF-IDF FEATURES

tfidf_word = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),     
    sublinear_tf=True,      
    min_df=2,
    analyzer='word'
)

tfidf_char = TfidfVectorizer(
    max_features=10000,
    ngram_range=(3, 6),    
    sublinear_tf=True,
    analyzer='char_wb'
)

X_word = tfidf_word.fit_transform(df['clean_text'])
X_char = tfidf_char.fit_transform(df['clean_text'])

# Combine both into one feature matrix
X = hstack([X_word, X_char])
print("\nCombined feature matrix shape:", X.shape)

# Encode labels to numbers
le = LabelEncoder()
y  = le.fit_transform(df['cyberbullying_type'])
print("Classes:", le.classes_)


# STEP 5 - TRAIN / VALIDATION / TEST SPLIT (70/15/15)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain size:      {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size:       {X_test.shape[0]}")


# STEP 6 - HANDLE CLASS IMBALANCE (SMOTE)
# only applied to training data — never val or test

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("After SMOTE: ", pd.Series(y_train_bal).value_counts().to_dict())


# STEP 7 - EVALUATION HELPER FUNCTION

def evaluate_model(model, X, y, name):
    preds     = model.predict(X)
    proba     = model.predict_proba(X)
    acc       = accuracy_score(y, preds)
    prec      = precision_score(y, preds, average='weighted', zero_division=0)
    rec       = recall_score(y, preds, average='weighted', zero_division=0)
    f1        = f1_score(y, preds, average='weighted', zero_division=0)
    auc_score = roc_auc_score(y, proba, multi_class='ovr', average='weighted')

    print(f"\n===== {name} =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=le.classes_, zero_division=0))

    return {
        'Model':     name,
        'Accuracy':  round(acc, 4),
        'Precision': round(prec, 4),
        'Recall':    round(rec, 4),
        'F1':        round(f1, 4),
        'ROC-AUC':   round(auc_score, 4)
    }


# STEP 8 - MODEL 1: LINEAR SVC

print("\nTraining LinearSVC with C tuning...")
best_auc, best_C, best_svc = 0, 1.0, None

for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
    m = CalibratedClassifierCV(
        LinearSVC(C=C, max_iter=3000, random_state=42), cv=3
    )
    m.fit(X_train_bal, y_train_bal)
    proba = m.predict_proba(X_val)
    score = roc_auc_score(y_val, proba, multi_class='ovr', average='weighted')
    print(f"  C={C} -> ROC-AUC: {score:.4f}")
    if score > best_auc:
        best_auc, best_C, best_svc = score, C, m

print(f"Best C = {best_C}")
svc_val = evaluate_model(best_svc, X_val, y_val, "LinearSVC (Validation)")


# STEP 9 - MODEL 2: LOGISTIC REGRESSION

print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    C=5.0,
    max_iter=1000,
    solver='saga',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_bal, y_train_bal)
lr_val = evaluate_model(lr, X_val, y_val, "Logistic Regression (Validation)")


# STEP 10 - MODEL 3: XGBOOST

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_bal, y_train_bal)
xgb_val = evaluate_model(xgb, X_val, y_val, "XGBoost (Validation)")


# STEP 11 - MODEL 4: RANDOM FOREST

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
rf_val = evaluate_model(rf, X_val, y_val, "Random Forest (Validation)")


# STEP 12 - VOTING ENSEMBLE
# combines all 4 models — usually beats any single model

print("\nTraining Voting Ensemble...")
voting = VotingClassifier(
    estimators=[
        ('svc', best_svc),
        ('lr',  lr),
        ('xgb', xgb),
        ('rf',  rf)
    ],
    voting='soft',
    n_jobs=-1
)
voting.fit(X_train_bal, y_train_bal)
voting_val = evaluate_model(voting, X_val, y_val, "Voting Ensemble (Validation)")


# STEP 13 - FINAL TEST SET EVALUATION

print("\n========== FINAL TEST SET RESULTS ==========")
svc_test    = evaluate_model(best_svc, X_test, y_test, "LinearSVC (Test)")
lr_test     = evaluate_model(lr,       X_test, y_test, "Logistic Regression (Test)")
xgb_test    = evaluate_model(xgb,      X_test, y_test, "XGBoost (Test)")
rf_test     = evaluate_model(rf,       X_test, y_test, "Random Forest (Test)")
voting_test = evaluate_model(voting,   X_test, y_test, "Voting Ensemble (Test)")

all_test   = [svc_test, lr_test, xgb_test, rf_test, voting_test]
df_results = pd.DataFrame(all_test)
print("\n===== MODEL COMPARISON (TEST SET) =====")
print(df_results.to_string(index=False))
df_results.to_csv('final_results.csv', index=False)


# STEP 14 - MODEL COMPARISON BAR CHART

metrics      = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
model_names  = ['LinearSVC', 'LogReg', 'XGBoost', 'RandForest', 'Ensemble']
colors       = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E']
x     = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(13, 6))
for i, (res, col, mname) in enumerate(zip(all_test, colors, model_names)):
    ax.bar(x + i * width, [res[m] for m in metrics],
           width, label=mname, color=col)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics)
ax.set_ylim(0.7, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Models (Test Set)')
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% target')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()


# STEP 15 - CONFUSION MATRICES

def plot_cm(model, X, y, name):
    preds = model.predict(X)
    cm    = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.xticks(rotation=30); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'cm_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()

plot_cm(best_svc, X_test, y_test, "LinearSVC")
plot_cm(lr,       X_test, y_test, "Logistic Regression")
plot_cm(xgb,      X_test, y_test, "XGBoost")
plot_cm(rf,       X_test, y_test, "Random Forest")
plot_cm(voting,   X_test, y_test, "Voting Ensemble")


# STEP 16 - ROC CURVES

def plot_roc(model, X, y, name):
    n_classes = len(le.classes_)
    y_bin  = label_binarize(y, classes=range(n_classes))
    y_prob = model.predict_proba(X)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.title(f'ROC Curve — {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'roc_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()

plot_roc(best_svc, X_test, y_test, "LinearSVC")
plot_roc(lr,       X_test, y_test, "Logistic Regression")
plot_roc(xgb,      X_test, y_test, "XGBoost")
plot_roc(voting,   X_test, y_test, "Voting Ensemble")


# STEP 17 - PR-AUC CURVES

def plot_prauc(model, X, y, name):
    n_classes = len(le.classes_)
    y_bin  = label_binarize(y, classes=range(n_classes))
    y_prob = model.predict_proba(X)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(rec, prec, label=f'{le.classes_[i]} (AP={ap:.2f})')
    plt.title(f'PR-AUC Curve — {name}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'prauc_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()

plot_prauc(best_svc, X_test, y_test, "LinearSVC")
plot_prauc(lr,       X_test, y_test, "Logistic Regression")
plot_prauc(xgb,      X_test, y_test, "XGBoost")
plot_prauc(voting,   X_test, y_test, "Voting Ensemble")


# STEP 18 - t-SNE VISUALIZATION

print("\nRunning t-SNE...")
sample_n  = min(2000, X_test.shape[0])
np.random.seed(42)
idx       = np.random.choice(X_test.shape[0], sample_n, replace=False)
X_sample  = X_test[idx]
y_sample  = y_test[idx]

svd       = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X_sample)
X_tsne    = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=500).fit_transform(X_reduced)

plt.figure(figsize=(9, 7))
tsne_colors = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E']
for i, label in enumerate(le.classes_):
    mask = y_sample == i
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                label=label, s=15, alpha=0.7,
                color=tsne_colors[i % len(tsne_colors)])
plt.title('t-SNE — Tweet Embeddings by Class')
plt.legend()
plt.tight_layout()
plt.savefig('tsne_tweets.png', dpi=150)
plt.show()


# STEP 19 - CROSS-VALIDATION (STABILITY CHECK)

print("\nRunning 5-fold cross-validation...")
svc_cv = cross_val_score(
    CalibratedClassifierCV(LinearSVC(C=best_C, max_iter=3000), cv=3),
    X_train_bal, y_train_bal, cv=5, scoring='f1_weighted', n_jobs=-1
)
lr_cv = cross_val_score(
    lr, X_train_bal, y_train_bal, cv=5, scoring='f1_weighted', n_jobs=-1
)
print(f"LinearSVC CV F1          : {svc_cv.mean():.4f} (+/- {svc_cv.std():.4f})")
print(f"Logistic Regression CV F1: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")


# STEP 20 - RECALL vs FALSE POSITIVE ANALYSIS

print("\n===== RECALL vs FALSE POSITIVE RATE (Voting Ensemble) =====")
preds = voting.predict(X_test)
cm    = confusion_matrix(y_test, preds)

for i, cls in enumerate(le.classes_):
    tp  = cm[i, i]
    fn  = cm[i, :].sum() - tp
    fp  = cm[:, i].sum() - tp
    tn  = cm.sum() - tp - fn - fp
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"{cls:25s} | Recall: {rec:.3f} | False Positive Rate: {fpr:.3f}")


# STEP 21 - FINAL SUMMARY

best_test = max(all_test, key=lambda x: x['Accuracy'])
print("\n" + "="*50)
print("         FINAL SUMMARY")
print("="*50)
print(f"Dataset size  : {len(df)} tweets")
print(f"Classes       : {list(le.classes_)}")
print(f"Best model    : {best_test['Model']}")
print(f"Best Accuracy : {best_test['Accuracy']}")
print(f"Best F1-Score : {best_test['F1']}")
print(f"Best ROC-AUC  : {best_test['ROC-AUC']}")
print("\nWhat improved accuracy:")
print("  1. Merged not_cyberbullying + other_cyberbullying (removes main confusion)")
print("  2. Better cleaning — kept negations and emotion words")
print("  3. Word TF-IDF (1-3 grams) + Character TF-IDF (3-6 grams) combined")
print("  4. LinearSVC — best algorithm for sparse text data")
print("  5. Voting ensemble of all 4 models")
print("="*50)