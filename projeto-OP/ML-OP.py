import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = {
    "Nome": ["Luffy", "Zoro", "Nami", "Usopp", "Sanji", "Robin", "Chopper", "Franky", "Brook", "Jinbe"],
    "Forca": [95, 90, 40, 35, 85, 50, 45, 70, 60, 80],
    "Inteligencia": [70, 60, 85, 75, 65, 90, 80, 70, 65, 60],
    "Velocidade": [85, 80, 65, 55, 80, 60, 50, 55, 70, 65],
    "Tipo_Fruta": ["Paramecia", "Nenhuma", "Nenhuma", "Nenhuma", "Nenhuma", "Paramecia", "Zoan", "Nenhuma", "Nenhuma", "Nenhuma"],
    "Habilidade": ["Haki", "Haki", "Clima-Tact", "Mira", "Haki", "Arqueologia", "Medicina", "Ciborgue", "Música", "Haki"],
    "Sobrevive": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Forca", "Inteligencia", "Velocidade", "Tipo_Fruta", "Habilidade"]]
y = df["Sobrevive"]

numeric_features = ["Forca", "Inteligencia", "Velocidade"]
categorical_features = ["Tipo_Fruta", "Habilidade"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42
)

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

melhor_modelo = None
melhor_score = 0

for nome, modelo in modelos.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("classifier", modelo)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    media = scores.mean()
    print(f"{nome} - Acurácia Média (CV): {media:.2f}")

    if media > melhor_score:
        melhor_score = media
        melhor_modelo = pipeline

melhor_modelo.fit(X_train, y_train)
joblib.dump(melhor_modelo, "onepiece_survival_model.pkl")

print("\n✅ Melhor modelo salvo como 'onepiece_survival_model.pkl'")