# ============================================================
# ÍNDICE
# ============================================================
#
# [1] IMPORTS
#     [1.1] Pandas
#     [1.2] NumPy
#     [1.3] SHAP Explainability
#     [1.4] XGBoost Classifier
#     [1.5] Train Test Split
#     [1.6] Probability Calibration
#     [1.7] ROC AUC Metric
#
# [2] CLASSE PredictiveLayer
#
#     [2.1] Inicialização da Classe
#         [2.1.1] __init__()
#
#     [2.2] Safe Helpers
#         [2.2.1] safe_numeric()
#         [2.2.2] column_exists()
#
#     [2.3] Feature Engineering
#         [2.3.1] prepare_features()
#         [2.3.1.1] Validação de Colunas
#         [2.3.1.2] Construção da Target
#         [2.3.1.3] Remoção de Colunas Não Utilizadas
#         [2.3.1.4] Segurança Numérica
#         [2.3.1.5] Customer Lifetime Value
#         [2.3.1.6] Flag Novo Cliente
#         [2.3.1.7] Flag Cliente High Value
#         [2.3.1.8] Conversão Categórica
#         [2.3.1.9] One Hot Encoding
#         [2.3.1.10] Limpeza de Nomes
#         [2.3.1.11] Remoção de NaN/INF
#         [2.3.1.12] Conversão Numérica Final
#         [2.3.1.13] Armazenamento das Features
#
#     [2.4] Treinamento do Modelo
#         [2.4.1] train_model()
#         [2.4.1.1] Preparação dos Dados
#         [2.4.1.2] Split Train/Test
#         [2.4.1.3] Balanceamento de Classes
#         [2.4.1.4] Modelo Base XGBoost
#         [2.4.1.5] Calibração de Probabilidade
#         [2.4.1.6] Predição no Teste
#         [2.4.1.7] Métrica ROC AUC
#         [2.4.1.8] Predição Dataset Completo
#         [2.4.1.9] Risk Level
#         [2.4.1.10] Retorno das Métricas
#
#     [2.5] Estimador Base
#         [2.5.1] get_base_estimator()
#
#     [2.6] Importância das Features
#         [2.6.1] get_feature_importance()
#         [2.6.1.1] Captura Importâncias
#         [2.6.1.2] Construção DataFrame
#         [2.6.1.3] Ordenação
#         [2.6.1.4] Limpeza de Nomes
#
#     [2.7] Clientes de Alto Risco
#         [2.7.1] get_high_risk_customers()
#
#     [2.8] Risco Executivo por Segmento
#         [2.8.1] get_executive_segment_risk()
#         [2.8.1.1] Construção do Segmento
#         [2.8.1.2] Agregação
#         [2.8.1.3] Receita Esperada em Risco
#         [2.8.1.4] Conversão Percentual
#
#     [2.9] Explainability SHAP
#         [2.9.1] generate_shap_explanations()
#         [2.9.1.1] Sample de Dados
#         [2.9.1.2] SHAP Explainer
#         [2.9.1.3] Compatibilidade de Versão
#         [2.9.1.4] Importância Global
#         [2.9.1.5] Segurança de Shape
#         [2.9.1.6] Construção DataFrame SHAP
#         [2.9.1.7] Limpeza de Nomes
#         [2.9.1.8] Fallback de Erro
#
#     [2.10] Dataset Scorado
#         [2.10.1] get_scored_dataset()
#
# ============================================================


# ============================================================
# [1] IMPORTS
# ============================================================

# ============================================================
# [1.1] Pandas
# Manipulação tabular
# ============================================================
import pandas as pd

# ============================================================
# [1.2] NumPy
# Operações numéricas
# ============================================================
import numpy as np

# ============================================================
# [1.3] SHAP Explainability
# Interpretabilidade do modelo
# ============================================================
import shap

# ============================================================
# [1.4] XGBoost Classifier
# Modelo preditivo principal
# ============================================================
from xgboost import XGBClassifier

# ============================================================
# [1.5] Train Test Split
# Separação treino/teste
# ============================================================
from sklearn.model_selection import train_test_split

# ============================================================
# [1.6] Probability Calibration
# Calibração probabilística
# ============================================================
from sklearn.calibration import (
    CalibratedClassifierCV
)

# ============================================================
# [1.7] ROC AUC Metric
# Métrica de performance
# ============================================================
from sklearn.metrics import (
    roc_auc_score
)


# ============================================================
# [2] CLASSE PredictiveLayer
# Camada preditiva de churn
# ============================================================
class PredictiveLayer:

    # ========================================================
    # [2.1] Inicialização da Classe
    # ========================================================

    # ========================================================
    # [2.1.1] __init__()
    # Inicializa variáveis internas
    # ========================================================
    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.model = None

        self.feature_columns = []

        self.X_processed = None

        self.shap_values = None

    # ========================================================
    # [2.2] Safe Helpers
    # ========================================================

    # ========================================================
    # [2.2.1] safe_numeric()
    # Conversão numérica segura
    # ========================================================
    @staticmethod
    def safe_numeric(series):

        return pd.to_numeric(
            series,
            errors="coerce"
        ).fillna(0)

    # ========================================================
    # [2.2.2] column_exists()
    # Verifica existência de coluna
    # ========================================================
    @staticmethod
    def column_exists(df, col):

        return col in df.columns

    # ========================================================
    # [2.3] Feature Engineering
    # ========================================================

    # ========================================================
    # [2.3.1] prepare_features()
    # Pipeline de preparação das features
    # ========================================================
    def prepare_features(self):

        df = self.df.copy()

        # ====================================================
        # [2.3.1.1] Validação de Colunas
        # ====================================================
        required_cols = [
            "Churn"
        ]

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Colunas obrigatórias ausentes: {missing}"
            )

        # ====================================================
        # [2.3.1.2] Construção da Target
        # ====================================================
        y = self.safe_numeric(
            df["Churn"]
        )

        # ====================================================
        # [2.3.1.3] Remoção de Colunas Não Utilizadas
        # ====================================================
        drop_cols = [
            "customerID",
            "Churn"
        ]

        X = df.drop(
            columns=drop_cols,
            errors="ignore"
        ).copy()

        # ====================================================
        # [2.3.1.4] Segurança Numérica
        # ====================================================
        numeric_candidates = [
            "MonthlyCharges",
            "tenure",
            "TotalCharges"
        ]

        for col in numeric_candidates:

            if self.column_exists(X, col):

                X[col] = self.safe_numeric(
                    X[col]
                )

        # ====================================================
        # [2.3.1.5] Customer Lifetime Value
        # ====================================================
        if (
            self.column_exists(X, "MonthlyCharges")
            and self.column_exists(X, "tenure")
        ):

            X["customer_lifetime_value"] = (
                X["MonthlyCharges"]
                * np.maximum(
                    X["tenure"],
                    1
                )
            )

        # ====================================================
        # [2.3.1.6] Flag Novo Cliente
        # ====================================================
        if self.column_exists(X, "tenure"):

            X["is_new_customer"] = (
                X["tenure"] <= 12
            ).astype(int)

        # ====================================================
        # [2.3.1.7] Flag Cliente High Value
        # ====================================================
        if self.column_exists(X, "MonthlyCharges"):

            X["high_value_customer"] = (
                X["MonthlyCharges"] >= 80
            ).astype(int)

        # ====================================================
        # [2.3.1.8] Conversão Categórica
        # ====================================================
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_cols:

            X[col] = (
                X[col]
                .astype(str)
                .fillna("Unknown")
            )

        # ====================================================
        # [2.3.1.9] One Hot Encoding
        # ====================================================
        X = pd.get_dummies(
            X,
            drop_first=True
        )

        # ====================================================
        # [2.3.1.10] Limpeza de Nomes
        # ====================================================
        X.columns = [

            str(col)

            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")

            for col in X.columns
        ]

        # ====================================================
        # [2.3.1.11] Remoção de NaN/INF
        # ====================================================
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0)

        # ====================================================
        # [2.3.1.12] Conversão Numérica Final
        # ====================================================
        for col in X.columns:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).fillna(0)

        # ====================================================
        # [2.3.1.13] Armazenamento das Features
        # ====================================================
        self.feature_columns = (
            X.columns.tolist()
        )

        return X, y
        # ========================================================
    # [2.4] Treinamento do Modelo
    # ========================================================

    # ========================================================
    # [2.4.1] train_model()
    # Treina modelo preditivo de churn
    # ========================================================
    def train_model(self):

        # ====================================================
        # [2.4.1.1] Preparação dos Dados
        # ====================================================
        X, y = self.prepare_features()

        self.X_processed = X.copy()

        # ====================================================
        # [2.4.1.2] Split Train/Test
        # ====================================================
        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,
                y,

                test_size=0.2,

                random_state=42,

                stratify=y
            )
        )

        # ====================================================
        # [2.4.1.3] Balanceamento de Classes
        # ====================================================
        negative = (y_train == 0).sum()

        positive = (y_train == 1).sum()

        scale_pos_weight = (
            negative / positive
            if positive > 0
            else 1
        )

        # ====================================================
        # [2.4.1.4] Modelo Base XGBoost
        # ====================================================
        base_model = XGBClassifier(

            n_estimators=250,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.85,

            colsample_bytree=0.85,

            objective="binary:logistic",

            eval_metric="logloss",

            random_state=42,

            scale_pos_weight=scale_pos_weight,

            min_child_weight=3,

            gamma=0.1,

            reg_alpha=0.2,

            reg_lambda=1.2,

            n_jobs=-1
        )

        # ====================================================
        # [2.4.1.5] Calibração de Probabilidade
        # ====================================================
        calibrated_model = (

            CalibratedClassifierCV(

                estimator=base_model,

                method="sigmoid",

                cv=3
            )
        )

        calibrated_model.fit(
            X_train,
            y_train
        )

        self.model = calibrated_model

        # ====================================================
        # [2.4.1.6] Predição no Teste
        # ====================================================
        probs = (

            calibrated_model
            .predict_proba(X_test)[:, 1]
        )

        # ====================================================
        # [2.4.1.7] Métrica ROC AUC
        # ====================================================
        auc = roc_auc_score(
            y_test,
            probs
        )

        # ====================================================
        # [2.4.1.8] Predição Dataset Completo
        # ====================================================
        full_probs = (

            calibrated_model
            .predict_proba(X)[:, 1]
        )

        self.df[
            "churn_probability"
        ] = full_probs

        # ====================================================
        # [2.4.1.9] Risk Level
        # ====================================================
        self.df["risk_level"] = pd.cut(

            self.df["churn_probability"],

            bins=[0, 0.3, 0.6, 1],

            labels=[
                "Low",
                "Medium",
                "High"
            ],

            include_lowest=True
        )

        # ====================================================
        # [2.4.1.10] Retorno das Métricas
        # ====================================================
        return {

            "auc":
                round(float(auc), 4),

            "avg_probability":
                round(
                    float(full_probs.mean()),
                    4
                ),

            "high_risk_customers":
                int(
                    (
                        self.df[
                            "risk_level"
                        ] == "High"
                    ).sum()
                )
        }

    # ========================================================
    # [2.5] Estimador Base
    # ========================================================

    # ========================================================
    # [2.5.1] get_base_estimator()
    # Recupera estimador calibrado
    # ========================================================
    def get_base_estimator(self):

        if self.model is None:

            return None

        try:

            return (
                self.model
                .calibrated_classifiers_[0]
                .estimator
            )

        except Exception:

            return None

    # ========================================================
    # [2.6] Importância das Features
    # ========================================================

    # ========================================================
    # [2.6.1] get_feature_importance()
    # Gera importância das variáveis
    # ========================================================
    def get_feature_importance(self):

        estimator = self.get_base_estimator()

        if estimator is None:

            return pd.DataFrame()

        # ====================================================
        # [2.6.1.1] Captura Importâncias
        # ====================================================
        importances = (
            estimator.feature_importances_
        )

        # ====================================================
        # [2.6.1.2] Construção DataFrame
        # ====================================================
        importance_df = pd.DataFrame({

            "feature":
                self.feature_columns,

            "importance":
                importances
        })

        # ====================================================
        # [2.6.1.3] Ordenação
        # ====================================================
        importance_df = (

            importance_df

            .sort_values(
                "importance",
                ascending=False
            )

            .reset_index(drop=True)
        )

        # ====================================================
        # [2.6.1.4] Limpeza de Nomes
        # ====================================================
        importance_df["feature"] = (

            importance_df["feature"]

            .str.replace("_", " ")
        )

        return importance_df

    # ========================================================
    # [2.7] Clientes de Alto Risco
    # ========================================================

    # ========================================================
    # [2.7.1] get_high_risk_customers()
    # Filtra clientes de alto risco
    # ========================================================
    def get_high_risk_customers(
        self,
        threshold: float = 0.7
    ):

        if (
            "churn_probability"
            not in self.df.columns
        ):

            return pd.DataFrame()

        high_risk = self.df[

            self.df[
                "churn_probability"
            ] >= threshold

        ].copy()

        return high_risk.sort_values(
            "churn_probability",
            ascending=False
        )
        # ========================================================
    # [2.8] Risco Executivo por Segmento
    # ========================================================

    # ========================================================
    # [2.8.1] get_executive_segment_risk()
    # Calcula risco agregado por segmento
    # ========================================================
    def get_executive_segment_risk(self):

        if (
            "churn_probability"
            not in self.df.columns
        ):

            return pd.DataFrame()

        temp_df = self.df.copy()

        # ====================================================
        # [2.8.1.1] Construção do Segmento
        # ====================================================
        if (
            "Contract" in temp_df.columns
            and "InternetService" in temp_df.columns
        ):

            temp_df["executive_segment"] = (

                temp_df["Contract"]
                .astype(str)

                + " | " +

                temp_df["InternetService"]
                .astype(str)
            )

        else:

            temp_df["executive_segment"] = (
                "Unknown Segment"
            )

        # ====================================================
        # [2.8.1.2] Agregação
        # ====================================================
        segment_risk = (

            temp_df

            .groupby("executive_segment")

            .agg(

                customers=(
                    "customerID",
                    "count"
                ),

                avg_churn_probability=(
                    "churn_probability",
                    "mean"
                ),

                avg_monthly_revenue=(
                    "MonthlyCharges",
                    "mean"
                )
            )

            .reset_index()
        )

        # ====================================================
        # [2.8.1.3] Receita Esperada em Risco
        # ====================================================
        segment_risk[
            "expected_revenue_risk"
        ] = (

            segment_risk[
                "customers"
            ]

            * segment_risk[
                "avg_monthly_revenue"
            ]

            * 12

            * (
                segment_risk[
                    "avg_churn_probability"
                ]
            )
        )

        # ====================================================
        # [2.8.1.4] Conversão Percentual
        # ====================================================
        segment_risk[
            "avg_churn_probability"
        ] = (
            segment_risk[
                "avg_churn_probability"
            ] * 100
        ).round(2)

        return segment_risk.sort_values(

            "avg_churn_probability",

            ascending=False
        )

    # ========================================================
    # [2.9] Explainability SHAP
    # ========================================================

    # ========================================================
    # [2.9.1] generate_shap_explanations()
    # Explicabilidade global com SHAP
    # ========================================================
    def generate_shap_explanations(self):

        estimator = self.get_base_estimator()

        if (
            estimator is None
            or self.X_processed is None
        ):

            return pd.DataFrame()

        try:

            # ================================================
            # [2.9.1.1] Sample de Dados
            # ================================================
            sample_size = min(
                500,
                len(self.X_processed)
            )

            X_sample = (

                self.X_processed

                .sample(
                    sample_size,
                    random_state=42
                )
            )

            # ================================================
            # [2.9.1.2] SHAP Explainer
            # ================================================
            explainer = shap.TreeExplainer(
                estimator
            )

            shap_values = explainer.shap_values(
                X_sample
            )

            # ================================================
            # [2.9.1.3] Compatibilidade de Versão
            # ================================================
            if isinstance(shap_values, list):

                shap_values = shap_values[1]

            shap_values = np.array(
                shap_values
            )

            # ================================================
            # [2.9.1.4] Importância Global
            # ================================================
            mean_abs_shap = np.abs(
                shap_values
            ).mean(axis=0)

            # ================================================
            # [2.9.1.5] Segurança de Shape
            # ================================================
            min_len = min(
                len(self.feature_columns),
                len(mean_abs_shap)
            )

            # ================================================
            # [2.9.1.6] Construção DataFrame SHAP
            # ================================================
            shap_df = pd.DataFrame({

                "feature":
                    self.feature_columns[:min_len],

                "importance":
                    mean_abs_shap[:min_len]
            })

            shap_df = (

                shap_df

                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                .fillna(0)

                .sort_values(
                    "importance",
                    ascending=False
                )

                .reset_index(drop=True)
            )

            # ================================================
            # [2.9.1.7] Limpeza de Nomes
            # ================================================
            shap_df["feature"] = (

                shap_df["feature"]

                .str.replace("_", " ")
            )

            self.shap_values = shap_df

            return shap_df

        # ================================================
        # [2.9.1.8] Fallback de Erro
        # ================================================
        except Exception as e:

            print(
                f"Erro SHAP: {e}"
            )

            return self.get_feature_importance()

    # ========================================================
    # [2.10] Dataset Scorado
    # ========================================================

    # ========================================================
    # [2.10.1] get_scored_dataset()
    # Retorna dataset final com scores
    # ========================================================
    def get_scored_dataset(self):

        return self.df.copy()
    