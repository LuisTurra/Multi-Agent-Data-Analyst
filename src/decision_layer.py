# =========================================================
# ÍNDICE
# =========================================================
#
# [1] IMPORTS
#   [1.1] pandas
#   [1.2] numpy
#
# [2] CLASSE DECISIONLAYER
#
#   [2.1] ASSUMPTIONS
#       [2.1.1] Taxa de Contato
#       [2.1.2] Taxa de Engajamento
#       [2.1.3] Taxa Base de Retenção
#       [2.1.4] Custo por Cliente
#       [2.1.5] Meses Médios de Retenção
#
#   [2.2] calculate_priority
#       [2.2.1] Normalização de Churn
#       [2.2.2] Normalização de Receita
#       [2.2.3] Normalização de Clientes
#       [2.2.4] Normalização de ROI
#       [2.2.5] Score Final
#       [2.2.6] Classificação de Prioridade
#
#   [2.3] recommend_actions
#       [2.3.1] Verificação de Dados Vazios
#       [2.3.2] Iteração dos Segmentos
#       [2.3.3] Extração de Métricas
#       [2.3.4] Motor de Estratégia
#       [2.3.5] Ajuste por Tenure
#       [2.3.6] Funil Operacional
#       [2.3.7] Cálculo Financeiro
#       [2.3.8] ROI
#       [2.3.9] Prioridade Executiva
#       [2.3.10] Regras de Segurança
#       [2.3.11] Construção das Recomendações
#
#   [2.4] simulate_targeted_campaign
#       [2.4.1] Verificação de Dados Vazios
#       [2.4.2] Seleção dos Segmentos
#       [2.4.3] Métricas Base
#       [2.4.4] Cenários Executivos
#       [2.4.5] Funil Operacional
#       [2.4.6] Simulação Financeira
#       [2.4.7] Complexidade Operacional
#       [2.4.8] Variabilidade do ROI
#       [2.4.9] Construção dos Resultados
#
#   [2.5] build_executive_summary
#       [2.5.1] Verificação de Dados Vazios
#       [2.5.2] Construção do Resumo Executivo
#       [2.5.3] Retorno do DataFrame
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
# [2] CLASSE DECISIONLAYER
# Camada de decisão e recomendações executivas
# =========================================================
class DecisionLayer:

    # =====================================================
    # [2.1] ASSUMPTIONS
    # Premissas operacionais da modelagem
    # =====================================================
    ASSUMPTIONS = {

        # =================================================
        # [2.1.1] Taxa de Contato
        # =================================================
        "contact_rate": 0.65,

        # =================================================
        # [2.1.2] Taxa de Engajamento
        # =================================================
        "engagement_rate": 0.40,

        # =================================================
        # [2.1.3] Taxa Base de Retenção
        # =================================================
        "base_retention_success_rate": 0.18,

        # =================================================
        # [2.1.4] Custo por Cliente
        # =================================================
        "campaign_cost_per_customer": 14.0,

        # =================================================
        # [2.1.5] Meses Médios de Retenção
        # =================================================
        "avg_retention_months": 8
    }

    # =====================================================
    # [2.2] calculate_priority
    # Cálculo executivo de prioridade
    # =====================================================
    @staticmethod
    def calculate_priority(
        churn_rate: float,
        revenue_risk: float,
        customers: int,
        roi: float
    ) -> str:

        # -------------------------------------------------
        # [2.2.1] Normalização de Churn
        # -------------------------------------------------
        churn_score = min(
            churn_rate / 100,
            1
        )

        # -------------------------------------------------
        # [2.2.2] Normalização de Receita
        # -------------------------------------------------
        revenue_score = min(
            revenue_risk / 500000,
            1
        )

        # -------------------------------------------------
        # [2.2.3] Normalização de Clientes
        # -------------------------------------------------
        customer_score = min(
            customers / 2000,
            1
        )

        # -------------------------------------------------
        # [2.2.4] Normalização de ROI
        # -------------------------------------------------
        roi_score = min(
            max(roi, 0) / 300,
            1
        )

        # -------------------------------------------------
        # [2.2.5] Score Final
        # -------------------------------------------------
        risk_score = (

            churn_score * 0.40 +

            revenue_score * 0.35 +

            customer_score * 0.10 +

            roi_score * 0.15
        )

        # -------------------------------------------------
        # [2.2.6] Classificação de Prioridade
        # -------------------------------------------------
        if risk_score >= 0.75:

            return "Crítica"

        elif risk_score >= 0.55:

            return "Alta"

        elif risk_score >= 0.35:

            return "Média"

        else:

            return "Baixa"

    # =====================================================
    # [2.3] recommend_actions
    # Motor principal de recomendações
    # =====================================================
    @staticmethod
    def recommend_actions(
        prioritized: pd.DataFrame,
        kpis: dict
    ) -> pd.DataFrame:

        # =================================================
        # [2.3.1] Verificação de Dados Vazios
        # =================================================
        if prioritized is None or prioritized.empty:

            return pd.DataFrame()

        recommendations = []

        # =================================================
        # [2.3.2] Iteração dos Segmentos
        # =================================================
        for _, row in prioritized.head(8).iterrows():

            # =================================================
            # [2.3.3] Extração de Métricas
            # =================================================
            churn_rate = float(
                row.get("churn_rate", 0)
            )

            revenue_risk = float(
                row.get(
                    "revenue_at_risk_annual",
                    0
                )
            )

            customers = int(
                row.get("size", 0)
            )

            segment = row.get(
                "executive_segment",
                "-"
            )

            avg_tenure = float(
                row.get(
                    "avg_tenure",
                    0
                )
            )

            arpu = float(
                row.get("arpu", 0)
            )

            # =================================================
            # [2.3.4] Motor de Estratégia
            # =================================================
            if churn_rate >= 45:

                action = (
                    "Campanha imediata de retenção "
                    "com contato humano prioritário "
                    "e oferta personalizada"
                )

                reduction_pct = 0.20

                cost_multiplier = 1.30

            elif churn_rate >= 30:

                action = (
                    "Estratégia direcionada de "
                    "fidelização com benefícios "
                    "e incentivo de permanência"
                )

                reduction_pct = 0.14

                cost_multiplier = 1.10

            elif churn_rate >= 15:

                action = (
                    "Campanha preventiva de "
                    "relacionamento e retenção"
                )

                reduction_pct = 0.09

                cost_multiplier = 1.00

            else:

                action = (
                    "Monitoramento passivo "
                    "com retenção seletiva"
                )

                reduction_pct = 0.03

                cost_multiplier = 0.70

            # =================================================
            # [2.3.5] Ajuste por Tenure
            # =================================================
            if avg_tenure <= 12:

                reduction_pct *= 1.15

            elif avg_tenure >= 36:

                reduction_pct *= 0.90

            assumptions = DecisionLayer.ASSUMPTIONS

            # =================================================
            # [2.3.6] Funil Operacional
            # =================================================
            contacted_customers = (

                customers

                * assumptions["contact_rate"]
            )

            engaged_customers = (

                contacted_customers

                * assumptions["engagement_rate"]
            )

            retained_customers = (

                engaged_customers

                * reduction_pct
            )

            # =================================================
            # [2.3.7] Cálculo Financeiro
            # =================================================
            protected_revenue = (

                retained_customers

                * arpu

                * assumptions["avg_retention_months"]
            )

            estimated_campaign_cost = (

                contacted_customers

                * assumptions[
                    "campaign_cost_per_customer"
                ]

                * cost_multiplier
            )

            net_gain = (
                protected_revenue
                - estimated_campaign_cost
            )

            # =================================================
            # [2.3.8] ROI
            # =================================================
            roi = (
                (
                    net_gain
                    / estimated_campaign_cost
                ) * 100
                if estimated_campaign_cost > 0
                else 0
            )

            # =================================================
            # [2.3.9] Prioridade Executiva
            # =================================================
            priority = (
                DecisionLayer.calculate_priority(
                    churn_rate,
                    revenue_risk,
                    customers,
                    roi
                )
            )

            # =================================================
            # [2.3.10] Regras de Segurança
            # =================================================
            if roi < 0:

                priority = "Baixa"

                action = (
                    "Monitoramento passivo "
                    "e retenção seletiva"
                )

            # =================================================
            # [2.3.11] Construção das Recomendações
            # =================================================
            recommendations.append({

                "Prioridade":
                    priority,

                "Segmento Executivo":
                    segment,

                "Clientes":
                    customers,

                "Taxa de Churn (%)":
                    round(churn_rate, 2),

                "Receita Anual em Risco":
                    round(revenue_risk, 2),

                "Ação Recomendada":
                    action,

                "Receita Protegida":
                    round(protected_revenue, 2),

                "Custo Estimado da Campanha":
                    round(
                        estimated_campaign_cost,
                        2
                    ),

                "Ganho Líquido":
                    round(net_gain, 2),

                "ROI Estimado (%)":
                    round(roi, 1)
            })

        return pd.DataFrame(
            recommendations
        )
        # =====================================================
    # [2.4] simulate_targeted_campaign
    # Simulação executiva de campanhas
    # =====================================================
    @staticmethod
    def simulate_targeted_campaign(
        prioritized: pd.DataFrame,
        top_n: int = 5,
        churn_reduction_pct: int = 10,
        total_revenue_at_risk: float = 0
    ) -> dict:

        # =================================================
        # [2.4.1] Verificação de Dados Vazios
        # =================================================
        if (
            prioritized is None
            or prioritized.empty
        ):

            return {}

        # =================================================
        # [2.4.2] Seleção dos Segmentos
        # =================================================
        top_segments = prioritized.head(top_n)

        assumptions = DecisionLayer.ASSUMPTIONS

        # =================================================
        # [2.4.3] Métricas Base
        # =================================================
        revenue_at_risk = float(
            total_revenue_at_risk
        )

        customers_targeted = int(
            top_segments["size"].sum()
        )

        avg_arpu = float(
            top_segments["arpu"].mean()
        )

        reduction_factor = (
            churn_reduction_pct / 100
        )

        # =================================================
        # [2.4.4] Cenários Executivos
        # =================================================
        SCENARIOS = {

            "🟢 Conservador": {

                "churn_reduction_pct": 5,

                "engagement_rate": 0.25,

                "retention_efficiency": 0.12,

                "contact_rate": 0.55
            },

            "🟡 Base": {

                "churn_reduction_pct": 10,

                "retention_efficiency": 0.18,

                "engagement_rate": 0.40,

                "contact_rate": 0.65
            },

            "🔴 Agressivo": {

                "churn_reduction_pct": 15,

                "retention_efficiency": 0.25,

                "engagement_rate": 0.55,

                "contact_rate": 0.80
            }
        }

        results = {}

        for scenario_name, scenario in SCENARIOS.items():

            reduction_factor = (
                scenario["churn_reduction_pct"] / 100
            )

            contact_rate = (
                scenario["contact_rate"]
            )

            engagement_rate = (
                scenario["engagement_rate"]
            )

            # =============================================
            # [2.4.5] Funil Operacional
            # =============================================
            contacted_customers = (

                customers_targeted

                * contact_rate
            )

            engaged_customers = (

                contacted_customers

                * engagement_rate
            )

            retained_customers = (

                engaged_customers

                * reduction_factor
            )

            # =============================================
            # [2.4.6] Simulação Financeira
            # =============================================
            revenue_saved = (

                retained_customers

                * avg_arpu

                * assumptions[
                    "avg_retention_months"
                ]
            )

            campaign_cost = (

                contacted_customers

                * assumptions[
                    "campaign_cost_per_customer"
                ]
            )

            net_gain = (
                revenue_saved
                - campaign_cost
            )

            roi = (
                (
                    net_gain
                    / campaign_cost
                ) * 100
                if campaign_cost > 0
                else 0
            )

            # =============================================
            # [2.4.7] Complexidade Operacional
            # =============================================
            if scenario["churn_reduction_pct"] <= 5:

                complexity = "Baixa"

                confidence = "Alta"

            elif scenario["churn_reduction_pct"] <= 10:

                complexity = "Média"

                confidence = "Alta"

            else:

                complexity = "Alta"

                confidence = "Média"

            # =============================================
            # [2.4.8] Variabilidade do ROI
            # =============================================
            if complexity == "Baixa":

                roi_variability = 8

            elif complexity == "Média":

                roi_variability = 12

            else:

                roi_variability = 20

            roi_min = max(
                roi - roi_variability,
                0
            )

            roi_max = roi + roi_variability

            # =============================================
            # [2.4.9] Construção dos Resultados
            # =============================================
            results[scenario_name] = {

                "strategy_name":
                    "Campanha de Retenção Direcionada",

                "strategy_scope":
                    f"Top {top_n} Segmentos de Risco",

                "target_reduction_pct":
                    scenario["churn_reduction_pct"],

                "operational_complexity":
                    complexity,

                "confidence_level":
                    confidence,

                "revenue_saved":
                    round(revenue_saved, 2),

                "campaign_cost":
                    round(campaign_cost, 2),

                "net_gain":
                    round(net_gain, 2),

                "display_roi":
                    round(min(roi, 250), 1),

                "roi":
                    round(roi, 1),

                "roi_uncertainty_band":
                    (
                        "± 8%"
                        if scenario["churn_reduction_pct"] <= 5
                        else (
                            "± 12%"
                            if scenario["churn_reduction_pct"] <= 10
                            else "± 18%"
                        )
                    ),

                "assumptions": {

                    "target_segments":
                        top_n,

                    "expected_churn_reduction":
                        reduction_factor,

                    "contact_rate":
                        contact_rate,

                    "engagement_rate":
                        engagement_rate,

                    "retention_efficiency":
                        scenario[
                            "retention_efficiency"
                        ],

                    "campaign_cost_ratio":
                        0.20,

                    "time_horizon_months":
                        12,

                    "campaign_cost_per_customer":
                        assumptions[
                            "campaign_cost_per_customer"
                        ],

                    "avg_retention_months":
                        assumptions[
                            "avg_retention_months"
                        ],

                    "assumptions_methodology":
                        (
                            "Simulações executivas "
                            "baseadas em cenários "
                            "conservador, base e agressivo "
                            "para campanhas de retenção CRM."
                        ),

                    "financial_model":
                        "crm_retention_scenario",

                    "baseline_revenue_at_risk":
                        round(
                            revenue_at_risk,
                            2
                        )
                },

                "roi_variability":
                    roi_variability,

                "roi_range":
                    f"{roi_min:.0f}% – {roi_max:.0f}%",

                "roi_min":
                    round(roi_min, 1),

                "roi_max":
                    round(roi_max, 1),
            }

        return results

    # =====================================================
    # [2.5] build_executive_summary
    # Construção da tabela executiva
    # =====================================================
    @staticmethod
    def build_executive_summary(
        recommendations: pd.DataFrame
    ) -> pd.DataFrame:

        # =================================================
        # [2.5.1] Verificação de Dados Vazios
        # =================================================
        if (
            recommendations is None
            or recommendations.empty
        ):

            return pd.DataFrame()

        # =================================================
        # [2.5.2] Construção do Resumo Executivo
        # =================================================
        summary = {

            "Total de Segmentos":
                len(recommendations),

            "Total de Clientes":
                int(
                    recommendations[
                        "Clientes"
                    ].sum()
                ),

            "Receita Total em Risco":
                round(
                    recommendations[
                        "Receita Anual em Risco"
                    ].sum(),
                    2
                ),

            "Receita Total Protegida":
                round(
                    recommendations[
                        "Receita Protegida"
                    ].sum(),
                    2
                ),

            "Ganho Líquido Total":
                round(
                    recommendations[
                        "Ganho Líquido"
                    ].sum(),
                    2
                ),

            "ROI Médio (%)":
                round(
                    recommendations[
                        "ROI Estimado (%)"
                    ].mean(),
                    1
                )
        }

        # =================================================
        # [2.5.3] Retorno do DataFrame
        # =================================================
        return pd.DataFrame([summary])