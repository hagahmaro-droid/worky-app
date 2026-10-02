import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CSV = 'hotel_bookings.csv'
MODEL_OUT = 'hotel_cancellation_model.pkl'

# Same core cleaning/feature-engineering approach used in the notebook.
df = pd.read_csv(CSV)
df = df.drop_duplicates()
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df = df[df['adr'] >= 0].copy()

# Features created in the notebook.
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_people'] = df['adults'] + df['children'] + df['babies']
df['total_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']
df['total_previous_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']

# The notebook uses these columns for classification.
df = df.drop(columns=['reservation_status', 'reservation_status_date',
                      'stays_in_weekend_nights', 'stays_in_week_nights'])

X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    'accuracy': float(accuracy_score(y_test, pred)),
    'precision': float(precision_score(y_test, pred)),
    'recall': float(recall_score(y_test, pred)),
    'f1': float(f1_score(y_test, pred)),
    'feature_columns': X.columns.tolist(),
}

joblib.dump({'model': model, 'metrics': metrics}, MODEL_OUT)
print('Saved:', MODEL_OUT)
print(metrics)
