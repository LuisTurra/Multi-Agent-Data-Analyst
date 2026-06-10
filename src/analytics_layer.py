# =========================================================
# ÍNDICE
# =========================================================
#
# [1] IMPORTS
#   [1.1] pandas
#   [1.2] numpy
#   [1.3] plotly.express
#   [1.4] translate_segment
#
# [2] CLASSE ANALYTICSLAYER
#
#   [2.1] __init__
#       [2.1.1] Cópia do DataFrame
#       [2.1.2] Validação de Colunas Obrigatórias
#       [2.1.3] Conversão de Tipos Numéricos
#       [2.1.4] Tratamento da Coluna Churn
#       [2.1.5] Criação do Segmento Executivo
#       [2.1.6] Criação do Segmento Detalhado
#
#   [2.2] _build_executive_segment
#       [2.2.1] Leitura de Contrato
#       [2.2.2] Leitura de Internet
#       [2.2.3] Mapeamento Executivo
#       [2.2.4] Retorno do Segmento
#
#   [2.3] kpi_agent
#       [2.3.1] Total de Clientes
#       [2.3.2] Taxa de Churn
#       [2.3.3] ARPU
#       [2.3.4] Clientes Perdidos
#       [2.3.5] Receita Perdida Histórica
#       [2.3.6] Receita em Risco
#       [2.3.7] Retorno dos KPIs
#
#   [2.4] segment_risk_agent
#       [2.4.1] Agrupamento por Segmento
#       [2.4.2] Métricas dos Segmentos
#       [2.4.3] Filtro de Segmentos
#       [2.4.4] Verificação de DataFrame Vazio
#       [2.4.5] Receita em Risco Preditiva
#       [2.4.6] Receita em Risco Tradicional
#       [2.4.7] Churn Bruto
#       [2.4.8] Churn Percentual
#       [2.4.9] Contribuição de Receita
#       [2.4.10] Ordenação Final
#
#   [2.5] prioritization_agent
#       [2.5.1] Verificação de Dados Vazios
#       [2.5.2] Função de Normalização
#       [2.5.3] Normalização de Churn
#       [2.5.4] Normalização de Receita
#       [2.5.5] Normalização de Tamanho
#       [2.5.6] Score de Prioridade
#       [2.5.7] Ordenação
#       [2.5.8] Ranking
#       [2.5.9] Classificação de Prioridade
#
#   [2.6] simulation_agent
#       [2.6.1] Verificação de Dados Vazios
#       [2.6.2] Seleção Top 5
#       [2.6.3] Receita Base em Risco
#       [2.6.4] Simulação de Cenários
#       [2.6.5] Receita Recuperada
#       [2.6.6] Custo da Campanha
#       [2.6.7] Ganho Líquido
#       [2.6.8] ROI
#       [2.6.9] Nova Taxa de Churn
#
#   [2.7] create_churn_pie
#       [2.7.1] Contagem de Churn
#       [2.7.2] Labels
#       [2.7.3] Construção do Gráfico
#
#   [2.8] create_risk_bar
#       [2.8.1] Verificação de Dados Vazios
#       [2.8.2] Seleção Top 8
#       [2.8.3] Tradução de Segmentos
#       [2.8.4] Construção do Gráfico
#       [2.8.5] Layout do Gráfico
#
#   [2.9] create_segment_heatmap
#       [2.9.1] Criação da Pivot Table
#       [2.9.2] Construção do Heatmap
#       [2.9.3] Ajuste de Layout
#
#   [2.10] detailed_segment_analysis
#       [2.10.1] Filtragem por Segmento Executivo
#       [2.10.2] Verificação de Dados Vazios
#       [2.10.3] Agrupamento Detalhado
#       [2.10.4] Conversão de Churn
#       [2.10.5] Ordenação Final
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
# [1.3] plotly.express
# Criação de gráficos
# =========================================================
import plotly.express as px

# =========================================================
# [1.4] translate_segment
# Tradução dos segmentos executivos
# =========================================================
from src.utils.translations import translate_segment


# =========================================================
# [2] CLASSE ANALYTICSLAYER
# Camada principal de analytics e métricas
# =========================================================
class AnalyticsLayer:

    # =====================================================
    # [2.1] __init__
    # Inicialização e preparação dos dados
    # =====================================================
    def __init__(
        self,
        df: pd.DataFrame
    ):

        # =================================================
        # [2.1.1] Cópia do DataFrame
        # Evita alteração do DataFrame original
        # =================================================
        self.df = df.copy()

        # =====================================================
        # [2.1.2] Validação de Colunas Obrigatórias
        # =====================================================
        required_columns = [

            'customerID',

            'Churn',

            'MonthlyCharges',

            'Contract',

            'InternetService',

            'tenure',

            'tenure_bucket'
        ]

        missing = [

            c for c in required_columns
            if c not in self.df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # =====================================================
        # [2.1.3] Conversão de Tipos Numéricos
        # =====================================================
        self.df['MonthlyCharges'] = pd.to_numeric(
            self.df['MonthlyCharges'],
            errors='coerce'
        ).fillna(0)

        self.df['tenure'] = pd.to_numeric(
            self.df['tenure'],
            errors='coerce'
        ).fillna(0)

        # =====================================================
        # [2.1.4] Tratamento da Coluna Churn
        # =====================================================
        if self.df['Churn'].dtype == object:

            self.df['Churn'] = (
                self.df['Churn']
                .astype(str)
                .str.lower()
                .map({
                    'yes': 1,
                    'no': 0
                })
            )

        self.df['Churn'] = pd.to_numeric(
            self.df['Churn'],
            errors='coerce'
        ).fillna(0)

        # =====================================================
        # [2.1.5] Criação do Segmento Executivo
        # =====================================================
        self.df['executive_segment'] = (

            self.df.apply(
                self._build_executive_segment,
                axis=1
            )
        )

        # =====================================================
        # [2.1.6] Criação do Segmento Detalhado
        # =====================================================
        self.df['detailed_segment'] = (

            self.df['Contract'].astype(str)

            + " | " +

            self.df['InternetService'].astype(str)

            + " | " +

            self.df['tenure_bucket'].astype(str)
        )

    # =========================================================
    # [2.2] _build_executive_segment
    # Criação do rótulo executivo
    # =========================================================
    def _build_executive_segment(
        self,
        row
    ):

        # =====================================================
        # [2.2.1] Leitura de Contrato
        # =====================================================
        contract = str(
            row.get(
                'Contract',
                'Unknown'
            )
        )

        # =====================================================
        # [2.2.2] Leitura de Internet
        # =====================================================
        internet = str(
            row.get(
                'InternetService',
                'Unknown'
            )
        )

        # =====================================================
        # [2.2.3] Mapeamento Executivo
        # =====================================================
        mapping = {

            ('Month-to-month', 'Fiber optic'):
                'High Risk Fiber Customers',

            ('Month-to-month', 'DSL'):
                'High Risk DSL Customers',

            ('One year', 'Fiber optic'):
                'Mid-Term Fiber Customers',

            ('One year', 'DSL'):
                'Mid-Term DSL Customers',

            ('Two year', 'Fiber optic'):
                'Loyal Fiber Customers',

            ('Two year', 'DSL'):
                'Loyal DSL Customers'
        }

        # =====================================================
        # [2.2.4] Retorno do Segmento
        # =====================================================
        return mapping.get(
            (contract, internet),
            f"{contract} | {internet}"
        )

    # =========================================================
    # [2.3] kpi_agent
    # Geração de KPIs principais
    # =========================================================
    def kpi_agent(self) -> dict:

        # =====================================================
        # [2.3.1] Total de Clientes
        # =====================================================
        total_customers = len(self.df)

        # =====================================================
        # [2.3.2] Taxa de Churn
        # =====================================================
        churn_rate = self.df['Churn'].mean() * 100

        # =====================================================
        # [2.3.3] ARPU
        # =====================================================
        arpu = self.df['MonthlyCharges'].mean()

        # =====================================================
        # [2.3.4] Clientes Perdidos
        # =====================================================
        churned_customers = int(
            self.df['Churn'].sum()
        )

        # =====================================================
        # [2.3.5] Receita Perdida Histórica
        # =====================================================
        historical_revenue_lost = (
            self.df[self.df['Churn'] == 1]['MonthlyCharges'] * 12
        ).sum()

        # =====================================================
        # [2.3.6] Receita em Risco
        # =====================================================
        if "churn_probability" in self.df.columns:

            revenue_at_risk_annual = (
                self.df['MonthlyCharges']
                * 12
                * self.df['churn_probability']
            ).sum()

        else:

            revenue_at_risk_annual = (
                self.df['MonthlyCharges']
                * 12
                * self.df['Churn']
            ).sum()

        # =====================================================
        # [2.3.7] Retorno dos KPIs
        # =====================================================
        return {

            "total_customers":
                int(total_customers),

            "churn_rate":
                round(churn_rate, 2),

            "arpu":
                round(arpu, 2),

            "revenue_at_risk_annual":
                round(revenue_at_risk_annual, 2),

            "historical_revenue_lost":
                round(historical_revenue_lost, 2),

            "avg_tenure":
                round(
                    self.df['tenure'].mean(),
                    1
                ),

            "churned_customers":
                churned_customers
        }

    # =========================================================
    # [2.4] segment_risk_agent
    # Análise de risco por segmento
    # =========================================================
    def segment_risk_agent(
        self
    ) -> pd.DataFrame:

        # =====================================================
        # [2.4.1] Agrupamento por Segmento
        # =====================================================
        grouped = self.df.groupby(
            ['executive_segment']
        )

        # =====================================================
        # [2.4.2] Métricas dos Segmentos
        # =====================================================
        segments = grouped.agg(

            size=(
                'customerID',
                'count'
            ),

            churn_rate=(
                'Churn',
                'mean'
            ),

            arpu=(
                'MonthlyCharges',
                'mean'
            ),

            avg_tenure=(
                'tenure',
                'mean'
            ),

            dominant_contract=(
                'Contract',
                lambda x: (
                    x.mode().iloc[0]
                    if not x.mode().empty
                    else "Unknown"
                )
            ),

            dominant_internet=(
                'InternetService',
                lambda x: (
                    x.mode().iloc[0]
                    if not x.mode().empty
                    else "Unknown"
                )
            )

        ).reset_index()

        # =====================================================
        # [2.4.3] Filtro de Segmentos
        # =====================================================
        segments = segments[
            segments['size'] >= 30
        ].copy()

        # =====================================================
        # [2.4.4] Verificação de DataFrame Vazio
        # =====================================================
        if segments.empty:

            return pd.DataFrame()

        # =====================================================
        # [2.4.5] Receita em Risco Preditiva
        # =====================================================
        if "churn_probability" in self.df.columns:

            risk_df = self.df.copy()

            risk_df["annual_risk"] = (
                risk_df["MonthlyCharges"]
                * 12
                * risk_df["churn_probability"]
            )

            risk_by_segment = (

                risk_df

                .groupby("executive_segment")[
                    "annual_risk"
                ]

                .sum()

                .reset_index()
            )

            segments = segments.merge(
                risk_by_segment,
                on="executive_segment",
                how="left"
            )

            segments.rename(
                columns={
                    "annual_risk":
                        "revenue_at_risk_annual"
                },
                inplace=True
            )

        # =====================================================
        # [2.4.6] Receita em Risco Tradicional
        # =====================================================
        else:

            segments['revenue_at_risk_annual'] = (
                segments['size']
                * segments['arpu']
                * 12
                * segments['churn_rate']
            )

        # =====================================================
        # [2.4.7] Churn Bruto
        # =====================================================
        segments['churn_rate_raw'] = (
            segments['churn_rate']
        )

        # =====================================================
        # [2.4.8] Churn Percentual
        # =====================================================
        segments['churn_rate'] = (

            segments['churn_rate']

            * 100
        ).round(2)

        total_risk = (
            segments[
                'revenue_at_risk_annual'
            ].sum()
        )

        # =====================================================
        # [2.4.9] Contribuição de Receita
        # =====================================================
        if total_risk > 0:

            segments[
                'revenue_contribution_pct'
            ] = (

                segments[
                    'revenue_at_risk_annual'
                ]

                / total_risk

                * 100
            ).round(2)

        else:

            segments[
                'revenue_contribution_pct'
            ] = 0

        # =====================================================
        # [2.4.10] Ordenação Final
        # =====================================================
        segments = segments.sort_values(
            'revenue_at_risk_annual',
            ascending=False
        )

        return segments.reset_index(
            drop=True
        )

    # =========================================================
    # [2.5] prioritization_agent
    # Priorização de segmentos
    # =========================================================
    def prioritization_agent(
        self,
        segments: pd.DataFrame
    ) -> pd.DataFrame:

        # =====================================================
        # [2.5.1] Verificação de Dados Vazios
        # =====================================================
        if (
            segments is None
            or segments.empty
        ):

            return pd.DataFrame()

        df = segments.copy()

        # =====================================================
        # [2.5.2] Função de Normalização
        # =====================================================
        def normalize(series):

            min_v = series.min()
            max_v = series.max()

            if max_v == min_v:

                return pd.Series(
                    [0.5] * len(series),
                    index=series.index
                )

            return (
                (series - min_v)
                /
                (max_v - min_v)
            )

        # =====================================================
        # [2.5.3] Normalização de Churn
        # =====================================================
        df['churn_norm'] = normalize(
            df['churn_rate']
        )

        # =====================================================
        # [2.5.4] Normalização de Receita
        # =====================================================
        df['risk_norm'] = normalize(
            df['revenue_at_risk_annual']
        )

        # =====================================================
        # [2.5.5] Normalização de Tamanho
        # =====================================================
        df['size_norm'] = normalize(
            df['size']
        )

        # =====================================================
        # [2.5.6] Score de Prioridade
        # =====================================================
        df['priority_score'] = (

            df['churn_norm'] * 0.35

            +

            df['risk_norm'] * 0.45

            +

            df['size_norm'] * 0.20
        )

        # =====================================================
        # [2.5.7] Ordenação
        # =====================================================
        df = df.sort_values(
            'priority_score',
            ascending=False
        ).reset_index(drop=True)

        # =====================================================
        # [2.5.8] Ranking
        # =====================================================
        df['rank'] = (
            df.index + 1
        )

        # =====================================================
        # [2.5.9] Classificação de Prioridade
        # =====================================================
        df['priority'] = pd.cut(

            df['priority_score'],

            bins=[
                -0.01,
                0.4,
                0.7,
                1.0
            ],

            labels=[
                'Medium',
                'High',
                'Critical'
            ]
        )

        return df

    # =========================================================
    # [2.6] simulation_agent
    # Simulação de cenários
    # =========================================================
    def simulation_agent(
        self,
        kpis: dict,
        segments: pd.DataFrame
    ):

        # =====================================================
        # [2.6.1] Verificação de Dados Vazios
        # =====================================================
        if (
            segments is None
            or segments.empty
        ):

            return {}

        # =====================================================
        # [2.6.2] Seleção Top 5
        # =====================================================
        top5 = segments.head(5)

        # =====================================================
        # [2.6.3] Receita Base em Risco
        # =====================================================
        base_risk = float(

            top5[
                'revenue_at_risk_annual'
            ].sum()
        )

        scenarios = {}

        # =====================================================
        # [2.6.4] Simulação de Cenários
        # =====================================================
        for r in [5, 10, 15]:

            reduction_factor = (
                r / 100
            )

            # =================================================
            # [2.6.5] Receita Recuperada
            # =================================================
            revenue_saved = (
                base_risk
                * reduction_factor
            )

            # =================================================
            # [2.6.6] Custo da Campanha
            # =================================================
            intervention_cost = (
                revenue_saved
                * 0.35
            )

            # =================================================
            # [2.6.7] Ganho Líquido
            # =================================================
            net_gain = (
                revenue_saved
                - intervention_cost
            )

            # =================================================
            # [2.6.8] ROI
            # =================================================
            roi = (
                (net_gain / intervention_cost)
                * 100
                if intervention_cost > 0
                else 0
            )

            # =================================================
            # [2.6.9] Nova Taxa de Churn
            # =================================================
            scenarios[
                f"{r}% redução nos Top 5 segmentos"
            ] = {

                "revenue_saved":
                    round(
                        revenue_saved,
                        2
                    ),

                "campaign_cost":
                    round(
                        intervention_cost,
                        2
                    ),

                "net_gain":
                    round(
                        net_gain,
                        2
                    ),

                "roi":
                    round(
                        roi,
                        1
                    ),

                "new_churn_rate":
                    round(
                        kpis['churn_rate']
                        * (1 - reduction_factor),
                        2
                    )
            }

        return scenarios
        # =========================================================
    # [2.7] create_churn_pie
    # Gráfico de pizza de churn
    # =========================================================
    def create_churn_pie(self):

        # =====================================================
        # [2.7.1] Contagem de Churn
        # =====================================================
        churn_counts = (

            self.df['Churn']

            .value_counts()

            .sort_index()
        )

        # =====================================================
        # [2.7.2] Labels
        # =====================================================
        labels = [
            'Não Churn',
            'Churn'
        ]

        # =====================================================
        # [2.7.3] Construção do Gráfico
        # =====================================================
        fig = px.pie(

            names=labels,

            values=churn_counts.values,

            title="Distribuição de Churn",

            color_discrete_sequence=[
                '#00cc96',
                '#ef553b'
            ]
        )

        fig.update_traces(
            textinfo='percent+label'
        )

        return fig

    # =========================================================
    # [2.8] create_risk_bar
    # Gráfico de barras de receita em risco
    # =========================================================
    def create_risk_bar(
        self,
        prioritized: pd.DataFrame
    ):

        # =====================================================
        # [2.8.1] Verificação de Dados Vazios
        # =====================================================
        if (
            prioritized is None
            or prioritized.empty
        ):

            return px.bar(
                title="No risk data available"
            )

        # =====================================================
        # [2.8.2] Seleção Top 8
        # =====================================================
        top = prioritized.head(8).copy()

        # =====================================================
        # [2.8.3] Tradução de Segmentos
        # =====================================================
        top["executive_segment_pt"] = (
            top["executive_segment"]
            .apply(translate_segment)
        )

        # =====================================================
        # [2.8.4] Construção do Gráfico
        # =====================================================
        fig = px.bar(

            top,

            x='revenue_at_risk_annual',

            y='executive_segment_pt',

            orientation='h',

            title=(
                "Receita em Risco "
                "por Segmento Executivo"
            ),

            labels={

                'revenue_at_risk_annual':
                    'Receita em Risco (Anual)',

                'executive_segment_pt':
                    'Segmento Executivo'
            },

            color='priority_score',

            color_continuous_scale='Reds'
        )

        # =====================================================
        # [2.8.5] Layout do Gráfico
        # =====================================================
        fig.update_layout(

                height=350,

                margin=dict(
                    l=220,
                    r=40,
                    t=60,
                    b=40
                ),

                yaxis=dict(
                    tickfont=dict(
                        size=11
                    ),
                    automargin=True
                )
            )

        return fig

    # =========================================================
    # [2.9] create_segment_heatmap
    # Heatmap de churn por contrato e tenure
    # =========================================================
    def create_segment_heatmap(self):

        # =====================================================
        # [2.9.1] Criação da Pivot Table
        # =====================================================
        pivot = (

            self.df.pivot_table(

                values='Churn',

                index='Contract',

                columns='tenure_bucket',

                aggfunc='mean'
            )

            * 100
        )

        # =====================================================
        # [2.9.2] Construção do Heatmap
        # =====================================================
        fig = px.imshow(

            pivot,

            text_auto=".1f",

            aspect="auto",

            color_continuous_scale='RdYlBu_r',

            title=(
                "Heatmap - Taxa de Churn (%) "
                "por Contrato e Tenure"
            )
        )

        # =====================================================
        # [2.9.3] Ajuste de Layout
        # =====================================================
        fig.update_layout(
            height=450
        )

        return fig

    # =========================================================
    # [2.10] detailed_segment_analysis
    # Drill-down detalhado por segmento
    # =========================================================
    def detailed_segment_analysis(
        self,
        executive_segment: str
    ) -> pd.DataFrame:

        # =====================================================
        # [2.10.1] Filtragem por Segmento Executivo
        # =====================================================
        subset = self.df[

            self.df['executive_segment']
            == executive_segment
        ]

        # =====================================================
        # [2.10.2] Verificação de Dados Vazios
        # =====================================================
        if subset.empty:

            return pd.DataFrame()

        # =====================================================
        # [2.10.3] Agrupamento Detalhado
        # =====================================================
        detailed = subset.groupby(
            ['detailed_segment']
        ).agg(

            size=(
                'customerID',
                'count'
            ),

            churn_rate=(
                'Churn',
                'mean'
            ),

            arpu=(
                'MonthlyCharges',
                'mean'
            )

        ).reset_index()

        # =====================================================
        # [2.10.4] Conversão de Churn
        # =====================================================
        detailed['churn_rate'] = (

            detailed['churn_rate']
            * 100
        ).round(2)

        # =====================================================
        # [2.10.5] Ordenação Final
        # =====================================================
        return detailed.sort_values(
            'churn_rate',
            ascending=False
        )