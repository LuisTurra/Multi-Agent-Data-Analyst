# =========================================================
# ÍNDICE
# =========================================================
#
# [1] IMPORTS
#   [1.1] pandas
#   [1.2] numpy
#   [1.3] BaseModel e ValidationError
#   [1.4] Tipagens
#
# [2] CLASSE CUSTOMERSCHEMA
#   [2.1] Campos do Cliente
#
# [3] CLASSE DATALAYER
#
#   [3.1] __init__
#       [3.1.1] Leitura do CSV
#       [3.1.2] Processamento Inicial
#
#   [3.2] _validate_schema
#       [3.2.1] Lista de Colunas Obrigatórias
#       [3.2.2] Verificação de Colunas Ausentes
#       [3.2.3] Conversão Segura de TotalCharges
#       [3.2.4] Remoção de Valores Inválidos
#
#   [3.3] _create_business_segments
#       [3.3.1] Cópia do DataFrame
#       [3.3.2] Criação de Buckets de Tenure
#       [3.3.3] Segmentação de Valor
#       [3.3.4] Segmentação de Risco
#       [3.3.5] Segmentação de Fidelidade
#
#   [3.4] _feature_engineering
#       [3.4.1] Cópia do DataFrame
#       [3.4.2] Métricas Financeiras
#       [3.4.3] Features de Risco
#       [3.4.4] Contagem de Serviços
#       [3.4.5] Flag de Alto Risco
#
#   [3.5] _process_data
#       [3.5.1] Cópia do DataFrame
#       [3.5.2] Validação
#       [3.5.3] Conversão de Churn
#       [3.5.4] Feature Engineering
#       [3.5.5] Segmentações de Negócio
#       [3.5.6] Criação do Segmento Completo
#
#   [3.6] get_data
#       [3.6.1] Retorno Seguro do DataFrame
#
#   [3.7] get_schema_info
#       [3.7.1] Informações Gerais
#       [3.7.2] Taxa de Churn
#       [3.7.3] Lista de Colunas
#       [3.7.4] Buckets de Tenure
#
# =========================================================


# =========================================================
# [1] IMPORTS
# =========================================================

# =========================================================
# [1.1] pandas
# Manipulação de DataFrames
# =========================================================
import pandas as pd

# =========================================================
# [1.2] numpy
# Operações numéricas
# =========================================================
import numpy as np

# =========================================================
# [1.3] BaseModel e ValidationError
# Validação de schema com Pydantic
# =========================================================
from pydantic import BaseModel, ValidationError

# =========================================================
# [1.4] Tipagens
# Tipagem de listas e dicionários
# =========================================================
from typing import List, Dict


# =========================================================
# [2] CLASSE CUSTOMERSCHEMA
# Schema de validação dos clientes
# =========================================================
class CustomerSchema(BaseModel):

    # =====================================================
    # [2.1] Campos do Cliente
    # Estrutura esperada do dataset
    # =====================================================
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    Churn: str


# =========================================================
# [3] CLASSE DATALAYER
# Camada principal de processamento dos dados
# =========================================================
class DataLayer:

    # =====================================================
    # [3.1] __init__
    # Inicialização da camada de dados
    # =====================================================
    def __init__(
        self,
        file_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ):

        # =================================================
        # [3.1.1] Leitura do CSV
        # =================================================
        self.raw_df = pd.read_csv(file_path)

        # =================================================
        # [3.1.2] Processamento Inicial
        # =================================================
        self.df = self._process_data()

    # =========================================================
    # [3.2] _validate_schema
    # Validação básica do schema
    # =========================================================
    def _validate_schema(
        self,
        df: pd.DataFrame
    ):

        # =====================================================
        # [3.2.1] Lista de Colunas Obrigatórias
        # =====================================================
        required_cols = [

            'customerID',
            'tenure',
            'MonthlyCharges',
            'TotalCharges',
            'Churn',
            'Contract',
            'InternetService',
            'PaymentMethod'
        ]

        # =====================================================
        # [3.2.2] Verificação de Colunas Ausentes
        # =====================================================
        missing = [

            col for col in required_cols
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Colunas obrigatórias ausentes: {missing}"
            )

        # =====================================================
        # [3.2.3] Conversão Segura de TotalCharges
        # =====================================================
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'],
            errors='coerce'
        )

        # =====================================================
        # [3.2.4] Remoção de Valores Inválidos
        # =====================================================
        df = df.dropna(
            subset=['TotalCharges']
        )

        return df

    # =========================================================
    # [3.3] _create_business_segments
    # Criação de segmentações de negócio
    # =========================================================
    def _create_business_segments(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        # =====================================================
        # [3.3.1] Cópia do DataFrame
        # =====================================================
        df = df.copy()

        # =====================================================
        # [3.3.2] Criação de Buckets de Tenure
        # =====================================================
        df['tenure_bucket'] = pd.cut(

            df['tenure'],

            bins=[
                0,
                6,
                12,
                24,
                36,
                60,
                np.inf
            ],

            labels=[
                '0-6 meses (Novo)',
                '7-12 meses',
                '13-24 meses',
                '25-36 meses',
                '37-60 meses',
                '60+ meses (Leal)'
            ],

            include_lowest=True
        )

        # =====================================================
        # [3.3.3] Segmentação de Valor
        # =====================================================
        df['value_segment'] = pd.qcut(

            df['MonthlyCharges'],

            q=3,

            labels=[
                'Low Value',
                'Medium Value',
                'High Value'
            ]
        )

        # =====================================================
        # [3.3.4] Segmentação de Risco
        # =====================================================
        df['risk_segment'] = np.where(

            (
                df['Contract']
                == 'Month-to-month'
            )

            &

            (
                df['tenure']
                <= 12
            ),

            'High Risk New',

            'Standard'
        )

        # =====================================================
        # [3.3.5] Segmentação de Fidelidade
        # =====================================================
        df['loyalty_segment'] = np.where(

            df['tenure'] >= 36,

            'Loyal',

            'At Risk'
        )

        return df

    # =========================================================
    # [3.4] _feature_engineering
    # Engenharia de features
    # =========================================================
    def _feature_engineering(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        # =====================================================
        # [3.4.1] Cópia do DataFrame
        # =====================================================
        df = df.copy()

        # =====================================================
        # [3.4.2] Métricas Financeiras
        # =====================================================
        df['monthly_revenue'] = (
            df['MonthlyCharges']
        )

        df['lifetime_revenue'] = (
            df['TotalCharges']
        )

        df['avg_monthly_revenue'] = (

            df['TotalCharges']

            /

            df['tenure'].replace(0, 1)
        )

        # =====================================================
        # [3.4.3] Features de Risco
        # =====================================================
        df['has_fiber'] = (
            df['InternetService']
            == 'Fiber optic'
        ).astype(int)

        df['month_to_month'] = (
            df['Contract']
            == 'Month-to-month'
        ).astype(int)

        df['electronic_check'] = (
            df['PaymentMethod']
            == 'Electronic check'
        ).astype(int)

        df['senior_citizen'] = (
            df['SeniorCitizen']
        )

        # =====================================================
        # [3.4.4] Contagem de Serviços
        # =====================================================
        service_cols = [

            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        df['num_services'] = df[
            service_cols
        ].apply(

            lambda x: sum(
                x.str.contains(
                    'Yes',
                    na=False
                )
            ),

            axis=1
        )

        # =====================================================
        # [3.4.5] Flag de Alto Risco
        # =====================================================
        df['high_risk'] = (

            (
                df['month_to_month']
                == 1
            )

            &

            (
                df['tenure']
                <= 12
            )

            &

            (
                df['MonthlyCharges']
                > 70
            )

        ).astype(int)

        return df

    # =========================================================
    # [3.5] _process_data
    # Pipeline principal de processamento
    # =========================================================
    def _process_data(
        self
    ) -> pd.DataFrame:

        # =====================================================
        # [3.5.1] Cópia do DataFrame
        # =====================================================
        df = self.raw_df.copy()

        # =====================================================
        # [3.5.2] Validação
        # =====================================================
        df = self._validate_schema(df)

        # =====================================================
        # [3.5.3] Conversão de Churn
        # =====================================================
        df['Churn'] = df['Churn'].map({

            'Yes': 1,

            'No': 0
        })

        # =====================================================
        # [3.5.4] Feature Engineering
        # =====================================================
        df = self._feature_engineering(df)

        # =====================================================
        # [3.5.5] Segmentações de Negócio
        # =====================================================
        df = self._create_business_segments(df)

        # =====================================================
        # [3.5.6] Criação do Segmento Completo
        # =====================================================
        df['segment_full'] = (

            df['Contract']

            + " | "

            + df['InternetService']

            + " | "

            + df['tenure_bucket'].astype(str)
        )

        return df

    # =========================================================
    # [3.6] get_data
    # Retorno seguro do DataFrame
    # =========================================================
    def get_data(
        self
    ) -> pd.DataFrame:

        # =====================================================
        # [3.6.1] Retorno Seguro do DataFrame
        # =====================================================
        return self.df.copy()

    # =========================================================
    # [3.7] get_schema_info
    # Informações gerais do dataset
    # =========================================================
    def get_schema_info(
        self
    ) -> Dict:

        # =====================================================
        # [3.7.1] Informações Gerais
        # [3.7.2] Taxa de Churn
        # [3.7.3] Lista de Colunas
        # [3.7.4] Buckets de Tenure
        # =====================================================
        return {

            "shape":
                self.df.shape,

            "churn_rate":
                round(
                    self.df['Churn'].mean() * 100,
                    2
                ),

            "columns":
                list(self.df.columns),

            "tenure_buckets":
                list(
                    self.df['tenure_bucket'].unique()
                )
        }