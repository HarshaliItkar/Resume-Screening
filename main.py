import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv("AI_Resume_Screening.csv")
df.columns = df.columns.str.strip()

text_col = "Skills"
num_cols = ["Experience (Years)", "Salary Expectation ($)", "Projects Count", "AI Score (0-100)"]
cat_cols = ["Education", "Job Role"]
target_col = "Recruiter Decision"

X_text = df[text_col].fillna("")
X_num = df[num_cols].fillna(0)
X_cat = df[cat_cols].fillna("Unknown")
y = df[target_col]



sbert = SentenceTransformer("all-MiniLM-L6-v2")
X_text_emb = sbert.encode(X_text.tolist(), show_progress_bar=True)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_ohe = ohe.fit_transform(X_cat)
X_final = np.hstack([X_text_emb, X_num_scaled, X_cat_ohe])



X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=42, stratify=y
)
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)


print("\nClassification Report:\n")
print(classification_report(y_test, clf.predict(X_test)))



text_feature_names = [f"emb_{i}" for i in range(X_text_emb.shape[1])]
num_feature_names = num_cols
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

feature_names = text_feature_names + num_feature_names + cat_feature_names

masker = shap.maskers.Independent(X_train)
explainer = shap.LinearExplainer(clf, masker=masker, feature_names=feature_names)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
plt.show()

idx = int(input(f"Select candidate index (0 to {len(X_test)-1}): "))

explanation = shap.Explanation(
    values=shap_values[idx].values,
    base_values=shap_values[idx].base_values,
    data=X_test[idx],
    feature_names=feature_names
)

plt.figure(figsize=(12,5))
shap.plots.waterfall(explanation)
plt.show()
