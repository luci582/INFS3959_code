
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,precision_recall_curve, roc_curve, auc)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from xgboost import plot_importance
import traceback

TRAINING_CSV_PATH = "https://raw.githubusercontent.com/luci582/dataforinfs3959/refs/heads/main/APA-DDoS-Dataset.csv"
LOG_FILE_TO_ANALYZE = "/Users/luci-/Downloads/INFS3959/others/APA-DDoS-Dataset2.csv" 
LOG_DELIMITER = ","
ddos_original = pd.read_csv(TRAINING_CSV_PATH)
ddos_original.info()

print(ddos_original.head().T)
print(ddos_original.isna().sum())
print(ddos_original.groupby('Label').size())
columns_to_drop = ['tcp.dstport', 'ip.proto', 'tcp.flags.syn', 'tcp.flags.reset','tcp.flags.ack', 'ip.flags.mf', 'ip.flags.rb', 'tcp.seq', 'tcp.ack','frame.time','ip.len','tcp.len'
]
existing_columns_to_drop = [col for col in columns_to_drop if col in ddos_original.columns]
print(f" dropped stuff  {existing_columns_to_drop}")
ddos_processed = ddos_original.drop(columns=existing_columns_to_drop).copy()
ddos_processed.info()
print(ddos_processed['Label'].isna().sum(),ddos_processed['Label'].isnull().sum())
print(ddos_processed['Label'].value_counts())
ddos_processed['Label_new'] = ddos_processed['Label'].apply(lambda x: 'Benign' if x == 'Benign' else 'DDoS')
ddos_processed.drop(columns=['Label'], inplace=True)
ddos_processed.rename(columns={'Label_new': 'Label'}, inplace=True)
print(ddos_processed['Label'].value_counts())
y = ddos_processed['Label']
X = ddos_processed.drop(columns=['Label']).copy()
print(X)
print(y)
print(X.shape)
print(y.shape)
print(list(X.columns))
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)

label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
ddos_label_encoded = label_mapping.get('DDoS', None)

print(label_mapping)


categorical_columns = ['ip.src', 'ip.dst']
categorical_columns = [col for col in categorical_columns if col in X.columns]
print(categorical_columns)

numerical_columns = list(X.columns.difference(categorical_columns))

preprocessor = ColumnTransformer(
    transformers=[
        ('wabalabadubdub', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough', 
    verbose_feature_names_out=False
)
print(preprocessor)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_processed = pipeline.fit_transform(X) 
print(X_processed)
print(X.shape)
print(X)

processed_feature_names = pipeline.get_feature_names_out()
print(processed_feature_names[:10])


X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X.index)
print(X_processed_df.head())
 


X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=120, stratify=y_encoded
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


print("Class distribution in y_train:", np.bincount(y_train))
print("Class distribution in y_test:", np.bincount(y_test))



decision_tree_model = DecisionTreeClassifier(random_state=120)
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree Confusion Matrix:\n", cm_dt)


rf_model = RandomForestClassifier(random_state=120, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)
 


print("\nTraining XGBoost...")
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")

print("\nDetailed Evaluation (XGBoost)")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("XGBoost Confusion Matrix:")
tn, fp, fn, tp = cm_xgb.ravel()
print(f"                 Predicted Benign | Predicted DDoS")
print(f"Actual Benign    {tn:15d} | {fp:15d}")
print(f"Actual DDoS      {fn:15d} | {tp:15d}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("XGBoost Confusion Matrix")
plt.show()

print("\nPlotting XGBoost ROC Curve...")
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


def parse_log_file(log_file_path, delimiter):
    parsed_data = []
    expected_columns_for_pipeline = [
        'ip.src',
        'ip.dst',
        'frame.len',
        'tcp.srcport',
        'tcp.flags.push',
        'ip.flags.df',
        'Packets',
        'Bytes',
        'Tx Packets',
        'Tx Bytes',
        'Rx Packets',
        'Rx Bytes'
    ]
    log_indices = {
        'ip.src': 0,
        'ip.dst': 1,
        'tcp.srcport': 2,
        'frame.len': 5,
        'tcp.flags.push': 8,
        'ip.flags.df': 11,
        'Packets': 16,
        'Bytes': 17,
        'Tx Packets': 18,
        'Tx Bytes': 19,
        'Rx Packets': 20,
        'Rx Bytes': 21
    }

    min_parts_needed = max(log_indices.values()) + 1
    processed_lines = 0
    skipped_lines = 0

    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                skipped_lines += 1
                continue
            try:
                parts = line.split(delimiter)
                if len(parts) >= min_parts_needed:
                    entry = {}
                    entry['ip.src'] = parts[log_indices['ip.src']]
                    entry['ip.dst'] = parts[log_indices['ip.dst']]
                    entry['frame.len'] = float(parts[log_indices['frame.len']])
                    entry['tcp.srcport'] = int(parts[log_indices['tcp.srcport']])
                    entry['tcp.flags.push'] = int(parts[log_indices['tcp.flags.push']])
                    entry['ip.flags.df'] = int(parts[log_indices['ip.flags.df']])
                    entry['Packets'] = float(parts[log_indices['Packets']])
                    entry['Bytes'] = float(parts[log_indices['Bytes']])
                    entry['Tx Packets'] = float(parts[log_indices['Tx Packets']])
                    entry['Tx Bytes'] = float(parts[log_indices['Tx Bytes']])
                    entry['Rx Packets'] = float(parts[log_indices['Rx Packets']])
                    entry['Rx Bytes'] = float(parts[log_indices['Rx Bytes']])
                    if all(key in entry for key in expected_columns_for_pipeline):
                         parsed_data.append(entry)
                         processed_lines += 1
                    else:
                         print(f"Warning: Internal logic error - not all expected keys populated for line {line_num+1}. Skipping.")
                         skipped_lines += 1
                else:
                     skipped_lines += 1
            except (ValueError, IndexError) as e:
                skipped_lines += 1
    print(f"\nLog file parsing finished.")
    print(f"  Successfully processed lines: {processed_lines}")
    print(f"  Skipped lines (empty/comment/malformed): {skipped_lines}")
    if not parsed_data:
         print("Warning: No valid data could be parsed from the log file matching the expected format.")
         return None
    log_df = pd.DataFrame(parsed_data)
    log_df = log_df[expected_columns_for_pipeline]

    print(f"\nSuccessfully created DataFrame from log data. Shape: {log_df.shape}")
    print("Sample of parsed log data (DataFrame):")
    print(log_df.head())
    return log_df




new_data_df = parse_log_file(LOG_FILE_TO_ANALYZE, LOG_DELIMITER)


if new_data_df is not None and not new_data_df.empty:
    X_log_processed = pipeline.transform(new_data_df)
    log_predictions = xgb_model.predict(X_log_processed)
    log_predictions_proba = xgb_model.predict_proba(X_log_processed)[:, ddos_label_encoded] 
    new_data_df['Prediction'] = log_predictions
    new_data_df['Prediction_Probability_DDoS'] = log_predictions_proba
    ddos_attacks = new_data_df[new_data_df['Prediction'] == ddos_label_encoded].copy()
    ddos_source_ips = ddos_attacks['ip.src'].unique()
    print("Log File Detection Results")
    if len(ddos_source_ips) > 0:
        print(f"Attack size {len(ddos_attacks)}")
        print(f"Found {len(ddos_source_ips)} unique source IP addresses")
        ip_counts = ddos_attacks['ip.src'].value_counts()
        print("\nSource IP Counts (Predicted as DDoS):")
        print(ip_counts)
    else:
        print("No DDoS activities detected in the provided log file based on the model's predictions.")