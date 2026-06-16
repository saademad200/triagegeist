```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. データの読み込み
train = pd.read_csv('/kaggle/input/competitions/triagegeist/train.csv')
test = pd.read_csv('/kaggle/input/competitions/triagegeist/test.csv')
print(train.shape)
train.head()
```


# トリアージはナースが行っているかな、自力で歩いて病院に入ってくる人と、警察？救急で来る人などに到着が分かれる、その他バイタルスコアがあり、結果がある、triage_actuityがターゲットか？


```python
print(test.shape)
test.head()  #　ちっとカラムが少ない
```


```python
# columnsを比較する
train_columns_list = list(train.columns)
test_columns_list = list(test.columns)

print('train columns', train_columns_list)
print('test_columns ', test_columns_list)
```


```python

# TrainにあってTestにないカラムを探す
target = 'triage_acuity'
train_cols = set(train.columns)
test_cols = set(test.columns)

# ターゲット以外の、Trainにしか存在しないカラムを抽出
leakage_candidates = list(train_cols - test_cols - {target})

print(f"--- Data Integrity Check ---")
print(f"Potential Leakage Columns detected: {leakage_candidates}")  # つまり、転帰と滞在時間がtestからは除外されている。リークの理由の一つ
```


```python
leakage_candidates
```


```python
# リークの根拠を示す分析（相関関係の可視化）
# 数値データのみで相関を確認
leakage_analysis = train[leakage_candidates + [target]].copy()
leakage_analysis.tail(10)
# dispositionはトリアージの結果なのでいらん
```


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

leakage_analysis["disposition_num"] = le.fit_transform(leakage_analysis["disposition"] )
leakage_analysis.drop('disposition', axis = 1).corr()
```


```python
leakage_analysis.drop('disposition', axis = 1).corr()
```


```python
# ed_los_hours（滞在時間）とターゲットの相関はかなりでかいので落とす。トリアージの結果である。（targetの内容と重複するcolumnである
# 臨床的な妥当性に基づいたカラム削除
# ed_los_hours: ED滞在時間はトリアージ後に確定する「未来の情報」
# disposition: 最終的な転帰（帰宅/入院）もトリアージ後の中身
cols_to_drop = leakage_candidates

print(f"\nAction: Dropping {cols_to_drop} to prevent Target Leakage.")
train_cleaned = train.drop(columns=cols_to_drop)
train_cleaned.head()
```


```python
# 5. 最終的な学習用データの確認
print(f"Cleaned Train shape: {train_cleaned.shape}")
print(f"Test shape: {test.shape}")
print(f"Fatures are now aligned (excluding target).")
```


```python
# 可視化（レポート用）
plt.figure(figsize=(10, 6))
sns.boxplot(x=target, y='ed_los_hours', data=train)
plt.title('Leakage Evidence: ED Length of Stay vs Triage Acuity')
plt.xlabel('Triage Acuity (1: Most Urgent, 5: Least Urgent)')
plt.ylabel('ED LOS (Hours)')
plt.savefig('leakage_evidence.png')
```

# 問題の特定(未来の情報、トリアージ後の情報を削除する)# 


* train.csv に含まれる ed_los_hours（滞在時間）と disposition（転帰）は、救急外来での**トリアージ判断が行われた後に決定される「未来の情報」**である


* 統計的根拠:分析の結果、ed_los_hours と triage_acuity の間には**強力な相関**がある。重症度が高い（スコアが小さい）ほど滞在時間が長くなるという医療現場の実態を反映。予測モデルには含めないことにする。


* 臨床的妥当性の確保 (Clinical Validity):リアルタイムの意思決定支援システム（Decision Support System）としての信頼性を担保するため、これらのリークカラムを意図的に削除し、**患者が到着した瞬間に**得られる情報（バイタル、主訴、病歴）のみを用いて学習



```python
patient_history = pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')
chief_complaints = pd.read_csv('/kaggle/input/competitions/triagegeist/chief_complaints.csv')
```


```python
patient_history.head()
```


```python
chief_complaints.head()
le = LabelEncoder()

chief_complaints['chief_complaint_system_number'] = le.fit_transform(chief_complaints['chief_complaint_system'])
chief_complaints
```


```python
# 2. 既往歴（Patient History）のマージ
# 全ての patient_id が一致する。 left join で結合
train_merged = train.merge(patient_history, on='patient_id', how='left')
test_merged = test.merge(patient_history, on='patient_id', how='left')

# 3. 主訴（Chief Complaints）のマージ
# 'chief_complaint_system' は既にあるため、'chief_complaint_raw'（テキスト）のみを抽出して結合
complaints_subset = chief_complaints[['patient_id', 'chief_complaint_raw']]
train_merged = train_merged.merge(complaints_subset, on='patient_id', how='left')
test_merged = test_merged.merge(complaints_subset, on='patient_id', how='left')

# 4. データリーク（未来情報）の削除
# トリアージ時点では知り得ない情報を train から削除します
leakage_cols = ['ed_los_hours', 'disposition']
train_merged = train_merged.drop(columns=leakage_cols)

# 結果の確認
print(f"学習データの最終カラム数: {len(train_merged.columns)}")
print(f"追加された既往歴の例: {patient_history.columns[1:5].tolist()}") # 最初の数件
```


```python
train_merged.head()
```


```python
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. モデルとトークナイザーの準備
# 医療系なので 'dmis-lab/biobert-v1.1' を使う
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```


```python

# GPUへ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

```


```python
def get_bert_embeddings(text_list, batch_size=32):
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]
        # トークン化
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=64, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS]トークンのベクトル（文章全体の意味を代表する）を取得
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    return np.vstack(all_embeddings)
```


```python
# 2. 実行（時間かかるぜ）
# train_merged['chief_complaint_raw'] を入力
texts = train_merged['chief_complaint_raw'].fillna("No Data").to_list()
bert_features = get_bert_embeddings(texts)

```


```python

# 3. 主成分分析(PCA)で次元圧縮（BERTの768次元は多すぎるため）
from sklearn.decomposition import PCA
pca = PCA(n_components=10) # 10次元程度に圧縮してモデルに入れやすくする
bert_pca = pca.fit_transform(bert_features)
```


```python
bert_pca
```


```python
# DataFrameに結合
bert_cols = [f'bert_pca_{i}' for i in range(10)]
print(bert_cols)
df_bert = pd.DataFrame(bert_pca, columns=bert_cols)
train_merged = pd.concat([train_merged, df_bert], axis=1)
```


```python
train_merged.head()
```


```python
# 1. データの準備
train = train_merged
target = 'triage_acuity'
```


```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
```


```python
# train = pd.concat([train, bert_df], axis=1)
train.head()
```


```python
# 特徴量の整理
# IDとテキスト原文、ターゲット、リークの恐れがあるものは除外
drop_cols = ['patient_id', 'triage_nurse_id', 'chief_complaint_raw', target]

```


```python
X = train.drop(columns=drop_cols)
y = train[target]

X.columns.values
# dropするcolumnを特定しdropする
```


```python
# カテゴリ変数の特定
cat_features = X.select_dtypes(include=['object']).columns.tolist()
cat_features
```


```python
# カテゴリデータを確かめる
X[cat_features]
```


```python
for col in X.columns:
    print(col)
```


```python

# 3. CatBoostによるクロスバリデーション
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

print("\nStarting Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost専用のデータセット形式
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=10000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    
    model.fit(train_pool, eval_set=val_pool)
    
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    cv_scores.append(score)
    models.append(model)
    print(f"Fold {fold+1} Accuracy: {score:.4f}")

print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
```


```python
import shap
import pandas as pd

# 1. SHAP値の計算 (学習済みの catboost モデルを使用)
# 注: model はクロスバリデーションで学習させたうちの1つ、または全データで再学習させたもの
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap_values
```


```python
shap_values.shape
```


```python
X_val.shape
```


```python
shap_values[:,:,0].shape
```


```python

# 3. 全体的な特徴量の重要度を可視化 (Summary Plot)
shap.summary_plot(shap_values[:,:,0], X_val) # 緊急度1に対する重要度
```


```python
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier, Pool
import torch
import matplotlib.pyplot as plt # Import matplotlib

# Re-run Cross-Validation to ensure variables are defined for SHAP explanation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

print("\nRe-running Cross-Validation to prepare for SHAP...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        early_stopping_rounds=50,
        verbose=0, # Suppress verbose output for re-run
        random_seed=42,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )

    model.fit(train_pool, eval_set=val_pool)

    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    cv_scores.append(score)
    models.append(model)

    # Save variables from the last fold for SHAP explanation
    if fold == kf.n_splits - 1:
        last_X_val = X_val
        last_preds = preds
        last_model = model
        last_target_names = ["1 (Resuscitation)", "2 (Emergent)", "3 (Urgent)", "4 (Less Urgent)", "5 (Non-Urgent)"]

print(f"Mean CV Accuracy from re-run: {np.mean(cv_scores):.4f}")

# Define the corrected explanation function
def explain_prediction_in_japanese(patient_index, X_data, shap_vals, preds, target_names):
    """
    特定の患者の予測根拠を日本語で説明する
    """
    p_idx_raw = preds[patient_index]
    pred_class_idx_original = int(p_idx_raw.flatten()[0]) # Original class label (1 to 5)
    pred_class_idx_0_indexed = pred_class_idx_original - 1 # 0-indexed for shap_vals access (0 to 4)

    print(f"DEBUG: preds[patient_index]: {p_idx_raw}")
    print(f"DEBUG: pred_class_idx_original: {pred_class_idx_original}")
    print(f"DEBUG: pred_class_idx_0_indexed: {pred_class_idx_0_indexed}")
    print(f"DEBUG: shap_vals type: {type(shap_vals)}")
    if isinstance(shap_vals, np.ndarray):
        print(f"DEBUG: shap_vals shape: {shap_vals.shape}")
    elif isinstance(shap_vals, list):
        print(f"DEBUG: shap_vals (list) length: {len(shap_vals)}")
        if len(shap_vals) > 0:
            print(f"DEBUG: shap_vals[0] shape: {shap_vals[0].shape}")


    # SHAP値の取り出し方: CatBoostClassifierのshap_valuesは (N_samples, N_features, N_classes) のndarrayを返すことが多い
    # または、各クラスに対する (N_samples, N_features) のndarrayのリストを返す。
    current_shap_vals = None
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # Case 1: 3D numpy array (N_samples, N_features, N_classes)
        # Check if the class index is within bounds for axis 2
        if pred_class_idx_0_indexed < shap_vals.shape[2]:
            current_shap_vals = shap_vals[patient_index, :, pred_class_idx_0_indexed]
        else:
            raise IndexError(f"Class index {pred_class_idx_0_indexed} out of bounds for shap_vals axis 2 with size {shap_vals.shape[2]}")
    elif isinstance(shap_vals, list) and len(shap_vals) > pred_class_idx_0_indexed:
        # Case 2: List of 2D numpy arrays (N_classes items, each (N_samples, N_features))
        current_shap_vals = shap_vals[pred_class_idx_0_indexed][patient_index]
    else:
        raise ValueError(f"Unexpected shape or type of shap_vals. Received type: {type(shap_vals)}, ndim: {getattr(shap_vals, 'ndim', 'N/A')}. Expected 3D numpy array or list of 2D arrays, with class index {pred_class_idx_0_indexed} accessible.")

    # Ensure current_shap_vals is not None before proceeding
    if current_shap_vals is None:
        raise ValueError("Failed to retrieve current_shap_vals for explanation.")

    # 特徴量名とSHAP値を確実に対応させる
    # Check lengths before creating DataFrame
    if len(X_data.columns.tolist()) != len(current_shap_vals.flatten()):
        print(f"DEBUG: X_data columns length: {len(X_data.columns.tolist())}")
        print(f"DEBUG: current_shap_vals flattened length: {len(current_shap_vals.flatten())}")
        raise ValueError("Mismatch in feature names and SHAP values length.")

    feature_importance = pd.DataFrame({
        'feature': X_data.columns.tolist(),
        'importance': current_shap_vals.flatten() # 確実に1次元配列にする
    })

    # 影響度の大きい順に並べ替え
    feature_importance_positive = feature_importance[feature_importance['importance'] > 0].sort_values(by='importance', ascending=False)
    feature_importance_negative = feature_importance[feature_importance['importance'] < 0].sort_values(by='importance', ascending=True) # 緊急度を下げる要因

    # 判定スコアの表示（緊急度は1始まりが多いので調整が必要な場合はここで行う）
    explanation = f"【判定結果】予測クラス: {target_names[pred_class_idx_0_indexed]}\n" # Adjust for 0-indexed target_names

    explanation += "\n【緊急度を上げている主な要因】\n"
    if not feature_importance_positive.empty:
        for i, row in feature_importance_positive.head(3).iterrows():
            # X_data から実際の値を取得（ilocを使用）
            val = X_data.iloc[patient_index][row['feature']]
            explanation += f"・{row['feature']}（値: {val}） → 寄与度: +{row['importance']:.3f}\n"
    else:
        explanation += "（特になし）\n"

    explanation += "\n【緊急度を下げている（または通常範囲に収めている）主な要因】\n"
    if not feature_importance_negative.empty:
        for i, row in feature_importance_negative.head(3).iterrows():
            val = X_data.iloc[patient_index][row['feature']]
            # マイナスの寄与度なので絶対値で表示し、緊急度を下げる方向であることを明記
            explanation += f"・{row['feature']}（値: {val}） → 寄与度: {row['importance']:.3f}\n"
    else:
        explanation += "（特になし）\n"

    return explanation

# Calculate SHAP values for the last fold's model and validation set
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(last_model)
# CatBoost's predict() returns 1-5, but SHAP's shap_values for MultiClass will be 0-indexed for classes.
# So, when accessing shap_vals, we need to subtract 1 from the predicted class label.
shap_values = explainer.shap_values(last_X_val)

# Execute the explanation function for the first patient in the last validation set
print("\n--- Individual Patient Explanation ---")
print(explain_prediction_in_japanese(0, last_X_val, shap_values, last_preds, last_target_names))

# Generate Summary Plot (Beeswarm)
print("\n--- SHAP Summary Plot (Beeswarm) ---")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, last_X_val, show=False)
plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png')
plt.show() # Display the plot in Colab

# Generate Summary Plot (Bar)
print("\n--- SHAP Summary Plot (Bar) ---")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, last_X_val, plot_type="bar", show=False)
plt.title("SHAP Bar Plot of Feature Importance")
plt.tight_layout()
plt.savefig('shap_importance_bar.png')
plt.show() # Display the plot in Colab
```

#### 総合的な考察

1.  **ターゲットリーケージの検出と除去（`ed_los_hours`, `disposition`）**
    -   `ed_los_hours`（滞在時間）と`disposition`（転帰）は、トリアージ判断後に確定する「未来の情報」であり、これらを予測モデルに含めると、モデルが実際には知り得ない情報を使って高精度を出してしまう（ターゲットリーケージ）ことが判明しました。
    -   特に`ed_los_hours`は`triage_acuity`と強い負の相関（-0.756）を示しており、この情報をモデルから排除することは、医療現場で患者到着時に利用可能な情報のみでトリアージを予測するという目的に合致し、モデルの堅牢性と現実的な適用可能性を大きく向上させました。この慎重な特徴量選択は、モデルが実世界で信頼される上で不可欠です。

2.  **BERT特徴量の導入と影響**
    -   `chief_complaint_raw`のような非構造化テキストデータは、そのままでは機械学習モデルに利用できませんが、BioBERTを用いることで患者の主訴を768次元の数値ベクトル（埋め込み）として表現しました。さらにPCAで10次元に圧縮し、モデルに組み込みました。
    -   BERT特徴量の導入は、主訴に含まれる重要な臨床情報をモデルが学習することを可能にし、特に複雑な症状や稀なケースにおいて、予測精度と解釈可能性に寄与する可能性があります。これにより、より包括的な患者状態の評価が可能になり、従来の構造化データだけでは捉えきれないニュアンスをモデルが理解できるようになります。

3.  **CatBoostモデルの選択理由と性能**
    -   CatBoostは、カテゴリ変数を事前にOne-Hotエンコーディングすることなく直接扱える点で非常に優れています。これにより、データ前処理の手間が省け、カテゴリカルな情報の欠落を防ぎ、モデルの解釈性も維持しやすいという利点があります。また、欠損値の扱いやGPUサポートも強化されており、大規模データセットにおいても効率的な学習が可能です。
    -   クロスバリデーション（5-Fold StratifiedKFold）の結果、平均CV Accuracyは0.9737と非常に高い値を示しました。これは、モデルが各トリアージ緊急度クラスを高い精度で分類できることを意味し、実用的なレベルでの性能を有していると言えます。

4.  **SHAP分析と医療現場の知見**
    -   SHAP分析により、個別の予測に対する特徴量の寄与度と、全体的な特徴量重要度が可視化されました。
    -   **個別予測の例**：患者0の予測クラス「5 (Non-Urgent)」において、`bert_pca_0`、`gcs_total`、`pain_score`が緊急度を上げる要因として挙げられ、`news2_score`、`bert_pca_1`、`pulse_pressure`が緊急度を下げる要因として示されました。これは、GCS（Glasgow Coma Scale）や痛みのスコアが患者の重症度評価に直結するという医療現場の知見と一致します。一方で、`NEWS2_score`が緊急度を下げる要因として機能しているのは、当該患者のNEWS2が低い値（例：2）であり、それが比較的軽症と判断される根拠になっていると解釈できます。
    -   **全体的な特徴量重要度**：SHAPサマリープロット（Beeswarm PlotとBar Plot）から、NEWS2スコア、年齢、バイタルサイン（心拍数、呼吸数、血圧）、そしてBERT特徴量が高い重要度を持つことが示唆されます。これらの特徴量は、従来の医療において患者の重症度を判断する上で不可欠な要素であり、モデルがこれらの特徴量を適切に学習していることを裏付けます。

5.  **モデルの医療現場での解釈可能性と実用性**
    -   SHAP分析による個別予測の説明は、医師や看護師がモデルの判断根拠を理解し、その結果を信頼する上で非常に有用です。例えば、「この患者はNEWS2スコアが低いから緊急度が低いと判断された」といった具体的な説明が可能になります。
    -   全体的な特徴量重要度も、モデルがどのような情報を重視しているかを明確にし、医療従事者がモデルの挙動を把握するのに役立ちます。これにより、モデルは単なる「ブラックボックス」ではなく、臨床意思決定支援ツールとして受け入れられやすくなります。
    -   高い予測精度と解釈可能性を兼ね備えることで、本モデルはトリアージの標準化、迅速化、さらには経験の浅い医療従事者の支援に貢献できる可能性があります。

6.  **データ作成段階での潜在的な課題や欠点**
    -   **データセットの代表性**：使用されたデータセットが特定の医療機関や地域に偏っていないか、多様な患者層を代表しているかという点は重要です。もし偏りがあれば、モデルの汎化性能が低下し、異なる環境での適用が困難になる可能性があります。
    -   **バイアスの可能性**：データ内の性別、人種、社会経済的地位などに関するバイアスが、モデルの予測にも反映される可能性があります。例えば、特定のグループの患者に対して不公平なトリアージ結果を出すリスクが考えられます。特に`language`や`insurance_type`のような特徴量は、社会的なバイアスを含みやすい要素です。
    -   **BERT埋め込みの限界**：BERTモデルは非常に強力ですが、医療テキスト特有の専門用語や省略形、あるいは文脈の複雑さを完全に捉えきれない場合があります。また、稀な疾患や特殊な主訴に対する表現学習には限界があるかもしれません。さらに、"No Data"などの欠損値補完は、情報が完全に失われている場合にノイズとなる可能性があります。
    -   **データ鮮度**：医療データは時間とともに変化するため、データが古くなるとモデルの性能が低下する可能性があります。定期的なデータ更新とモデルの再学習が不可欠です。

7.  **主要な知見と結論**
    -   本分析では、ターゲットリーケージの厳格な排除、BioBERTを用いたテキストデータの効果的な活用、そしてCatBoostの高い予測性能と解釈性という3つの主要な柱に基づき、高精度かつ説明可能なトリアージ予測モデルを構築しました。
    -   モデルは平均CV Accuracy 0.9737を達成し、SHAP分析により、NEWS2スコアやGCS、バイタルサインといった臨床的に重要な特徴量が予測に大きく寄与していることが確認されました。
    -   これにより、本モデルは医療現場におけるトリアージ支援ツールとして高い潜在能力を持つことが示されました。個別の予測根拠を説明できる能力は、医療従事者の信頼を得て、実際の臨床意思決定に統合されるための鍵となります。
    -   今後の改善点としては、より多様なデータソースの統合によるデータセットの代表性の向上、バイアス検出と軽減のための追加分析、そして継続的なモデルの性能監視と再学習が挙げられます。これらの課題に取り組むことで、より公平で堅牢な、実用的なAI医療支援システムの実現に繋がるでしょう。

