# =========================================================
# ÍNDICE
# =========================================================
#
# [1] IMPORTS
#   [1.1] Groq
#   [1.2] os
#   [1.3] json
#   [1.4] typing
#   [1.5] pandas
#   [1.6] translate_segment
#
# [2] CLASSE NARRATIVELAYER
#
#   [2.1] __init__
#       [2.1.1] Leitura da API Key
#       [2.1.2] Validação da API Key
#       [2.1.3] Inicialização do Cliente Groq
#
#   [2.2] safe_value
#       [2.2.1] Tratamento de Valores Nulos
#
#   [2.3] safe_float
#       [2.3.1] Conversão Segura para Float
#       [2.3.2] Fallback de Segurança
#
#   [2.4] validate_required_columns
#       [2.4.1] Verificação de Colunas Obrigatórias
#       [2.4.2] Erro de Schema
#
#   [2.5] sanitize_records
#       [2.5.1] Limpeza de Valores Nulos
#       [2.5.2] Construção de Registros Limpos
#
#   [2.6] build_executive_facts
#       [2.6.1] Inicialização dos Insights
#       [2.6.2] Concentração de Receita em Risco
#       [2.6.3] Segmento Executivo Principal
#       [2.6.4] Análise de Tenure
#       [2.6.5] Contexto Executivo de Churn
#       [2.6.6] Fatos de Simulação Financeira
#
#   [2.7] generate
#       [2.7.1] Sanitização dos KPIs
#       [2.7.2] Validação dos Segmentos
#       [2.7.3] Limpeza dos Segmentos
#       [2.7.4] Validação das Recomendações
#       [2.7.5] Limpeza das Recomendações
#       [2.7.6] Seleção do Cenário Base
#       [2.7.7] Sanitização das Simulações
#       [2.7.8] Construção dos Fatos Executivos
#       [2.7.9] System Prompt
#       [2.7.10] User Prompt
#       [2.7.11] Chamada da API Groq
#       [2.7.12] Pós-processamento da Narrativa
#       [2.7.13] Tratamento de Resposta Vazia
#       [2.7.14] Tratamento de Exceções
#
# =========================================================


# =========================================================
# [1] IMPORTS
# =========================================================

# =========================================================
# [1.1] Groq
# Cliente da API Groq
# =========================================================
from groq import Groq

# =========================================================
# [1.2] os
# Variáveis de ambiente
# =========================================================
import os

# =========================================================
# [1.3] json
# Serialização JSON
# =========================================================
import json

# =========================================================
# [1.4] typing
# Tipagem estática
# =========================================================
from typing import Dict, Any

# =========================================================
# [1.5] pandas
# Manipulação de DataFrames
# =========================================================
import pandas as pd

# =========================================================
# [1.6] translate_segment
# Tradução de segmentos executivos
# =========================================================
from src.utils.translations import (
    translate_segment
)


# =========================================================
# [2] CLASSE NARRATIVELAYER
# Camada de geração narrativa executiva
# =========================================================
class NarrativeLayer:

    # =====================================================
    # [2.1] __init__
    # Inicialização do cliente Groq
    # =====================================================
    def __init__(self):

        # =================================================
        # [2.1.1] Leitura da API Key
        # =================================================
        api_key = os.getenv("GROQ_API_KEY")

        # =================================================
        # [2.1.2] Validação da API Key
        # =================================================
        if not api_key:

            raise ValueError(
                "GROQ_API_KEY não encontrada "
                "nas variáveis de ambiente."
            )

        # =================================================
        # [2.1.3] Inicialização do Cliente Groq
        # =================================================
        self.client = Groq(api_key=api_key)

    # =====================================================
    # [2.2] safe_value
    # Tratamento seguro de valores nulos
    # =====================================================
    def safe_value(self, value, default=0):

        # =================================================
        # [2.2.1] Tratamento de Valores Nulos
        # =================================================
        if pd.isna(value) or value is None:

            return default

        return value

    # =====================================================
    # [2.3] safe_float
    # Conversão segura para float
    # =====================================================
    def safe_float(self, value, default=0.0):

        if pd.isna(value) or value is None:

            return default

        # =================================================
        # [2.3.1] Conversão Segura para Float
        # =================================================
        try:

            return float(value)

        # =================================================
        # [2.3.2] Fallback de Segurança
        # =================================================
        except Exception:

            return default

    # =====================================================
    # [2.4] validate_required_columns
    # Validação de colunas obrigatórias
    # =====================================================
    def validate_required_columns(
        self,
        df: pd.DataFrame,
        columns: list,
        df_name: str
    ):

        # =================================================
        # [2.4.1] Verificação de Colunas Obrigatórias
        # =================================================
        missing = [

            col for col in columns
            if col not in df.columns
        ]

        # =================================================
        # [2.4.2] Erro de Schema
        # =================================================
        if missing:

            raise ValueError(
                f"{df_name} está faltando colunas obrigatórias: "
                f"{missing}"
            )

    # =====================================================
    # [2.5] sanitize_records
    # Limpeza de registros
    # =====================================================
    def sanitize_records(self, records):

        clean_records = []

        for row in records:

            clean_row = {}

            for k, v in row.items():

                # =========================================
                # [2.5.1] Limpeza de Valores Nulos
                # =========================================
                if pd.isna(v) or v is None:

                    clean_row[k] = 0

                else:

                    clean_row[k] = v

            # =============================================
            # [2.5.2] Construção de Registros Limpos
            # =============================================
            clean_records.append(clean_row)

        return clean_records

    # =====================================================
    # [2.6] build_executive_facts
    # Construção de insights executivos
    # =====================================================
    def build_executive_facts(
        self,
        kpis,
        top_segments,
        simulations
    ):

        # =================================================
        # [2.6.1] Inicialização dos Insights
        # =================================================
        facts = []

        # =================================================
        # [2.6.2] Concentração de Receita em Risco
        # =================================================
        total_risk = float(
            kpis.get(
                "revenue_at_risk_annual",
                0
            )
        )

        top3_risk = (
            top_segments
            .head(3)["revenue_at_risk_annual"]
            .sum()
        )

        concentration = (
            (top3_risk / total_risk) * 100
            if total_risk > 0 else 0
        )

        facts.append({

            "type":
                "risk_concentration",

            "description":
                (
                    f"Os 3 principais segmentos "
                    f"representam {concentration:.1f}% "
                    f"da receita anual em risco."
                )
        })

        # =================================================
        # [2.6.3] Segmento Executivo Principal
        # =================================================
        if len(top_segments) > 0:

            top = top_segments.iloc[0]

            facts.append({

                "type":
                    "top_segment",

                "description":
                    (
                        f"O principal segmento executivo "
                        f"de risco possui "
                        f"{int(top.get('size', 0)):,} clientes, "
                        f"taxa de churn de "
                        f"{top.get('churn_rate', 0):.1f}% "
                        f"e receita anual em risco de "
                        f"US$ {top.get('revenue_at_risk_annual', 0):,.0f}."
                    )
            })
                        # =============================================
            # [2.6.4] Análise de Tenure
            # =============================================
            avg_tenure = float(
                top.get(
                    "avg_tenure",
                    0
                )
            )

            if avg_tenure <= 12:

                facts.append({

                    "type":
                        "tenure_pattern",

                    "description":
                        (
                            "O comportamento observado "
                            "sugere maior vulnerabilidade "
                            "nos primeiros meses do ciclo "
                            "de vida do cliente."
                        )
                })

            elif avg_tenure <= 24:

                facts.append({

                    "type":
                        "tenure_pattern",

                    "description":
                        (
                            "O churn apresenta maior "
                            "concentração na fase "
                            "intermediária do relacionamento."
                        )
                })

            else:

                facts.append({

                    "type":
                        "tenure_pattern",

                    "description":
                        (
                            "O churn impacta clientes "
                            "mais maduros, sugerindo "
                            "possível desgaste de experiência "
                            "ou percepção de valor."
                        )
                })

        # =================================================
        # [2.6.5] Contexto Executivo de Churn
        # =================================================
        churn_rate = float(
            kpis.get(
                "churn_rate",
                0
            )
        )

        if churn_rate >= 40:

            facts.append({

                "type":
                    "churn_context",

                "description":
                    (
                        "O churn atual representa "
                        "um cenário crítico de retenção."
                    )
            })

        elif churn_rate >= 25:

            facts.append({

                "type":
                    "churn_context",

                "description":
                    (
                        "O churn atual pressiona "
                        "significativamente a estabilidade "
                        "de receita recorrente."
                    )
            })

        # =================================================
        # [2.6.6] Fatos de Simulação Financeira
        # =================================================
        if simulations:

            revenue_saved = simulations.get(
                "revenue_saved",
                0
            )

            roi = simulations.get(
                "roi",
                0
            )

            confidence = simulations.get(
                "confidence_level",
                "Medium"
            )

            reduction = simulations.get(
                "target_reduction_pct",
                0
            )

            assumptions = simulations.get(
                "assumptions",
                {}
            )

            retention_rate = assumptions.get(
                "retention_success_rate",
                0
            )

            campaign_cost_pct = assumptions.get(
                "campaign_cost_pct",
                0
            )

            facts.append({

                "type":
                    "simulation",

                "description":
                    (
                        f"A simulação considera uma taxa "
                        f"de retenção efetiva de "
                        f"{retention_rate:.0%}, "
                        f"com custo operacional estimado "
                        f"em {campaign_cost_pct:.0%} "
                        f"da receita protegida. "
                        f"Sob essas premissas, "
                        f"uma redução de {reduction}% "
                        f"no churn dos segmentos prioritários "
                        f"poderia preservar aproximadamente "
                        f"US$ {revenue_saved:,.0f} "
                        f"em receita anual, "
                        f"com ROI estimado de "
                        f"{roi:.0f}%."
                    )
            })

            facts.append({

                "type":
                    "financial_assumptions",

                "description":
                    (
                        "As projeções financeiras consideram "
                        "premissas operacionais de contato, "
                        "engajamento de clientes, retenção "
                        "efetiva e custos médios de campanha."
                    )
            })

        return facts

    # =====================================================
    # [2.7] generate
    # Geração da narrativa executiva
    # =====================================================
    def generate(
        self,
        kpis: dict,
        top_segments: pd.DataFrame,
        simulations: dict,
        recommendations: pd.DataFrame
    ) -> str:

        try:

            # =================================================
            # [2.7.1] Sanitização dos KPIs
            # =================================================
            safe_kpis = {

                "total_customers":
                    self.safe_value(
                        kpis.get(
                            "total_customers"
                        ),
                        0
                    ),

                "churn_rate":
                    self.safe_float(
                        kpis.get(
                            "churn_rate"
                        ),
                        0
                    ),

                "arpu":
                    self.safe_float(
                        kpis.get(
                            "arpu"
                        ),
                        0
                    ),

                "revenue_at_risk_annual":
                    self.safe_float(
                        kpis.get(
                            "revenue_at_risk_annual"
                        ),
                        0
                    ),

                "churned_customers":
                    self.safe_value(
                        kpis.get(
                            "churned_customers"
                        ),
                        0
                    )
            }

            # =================================================
            # [2.7.2] Validação dos Segmentos
            # =================================================
            segment_columns = [

                "executive_segment",
                "size",
                "churn_rate",
                "arpu",
                "avg_tenure",
                "revenue_at_risk_annual",
                "priority",
                "revenue_contribution_pct"
            ]

            self.validate_required_columns(
                top_segments,
                segment_columns,
                "top_segments"
            )

            # =================================================
            # [2.7.3] Limpeza dos Segmentos
            # =================================================
            top_segments_clean = (
                top_segments[
                    segment_columns
                ]
                .head(5)
                .fillna(0)
            )

            segments_list = (
                self.sanitize_records(
                    top_segments_clean.to_dict(
                        orient="records"
                    )
                )
            )

            # =================================================
            # [2.7.4] Validação das Recomendações
            # =================================================
            rec_columns = [

                "Prioridade",
                "Segmento Executivo",
                "Ação Recomendada",
                "Clientes",
                "Taxa de Churn (%)",
                "Receita Anual em Risco",
                "Custo Estimado da Campanha",
                "Receita Protegida",
                "Ganho Líquido",
                "ROI Estimado (%)"
            ]

            self.validate_required_columns(
                recommendations,
                rec_columns,
                "recommendations"
            )

            # =================================================
            # [2.7.5] Limpeza das Recomendações
            # =================================================
            recs_clean_df = (
                recommendations
                .copy()
                .head(5)
                .fillna(0)
            )

            recs_clean = (
                self.sanitize_records(
                    recs_clean_df.to_dict(
                        orient="records"
                    )
                )
            )
                        # =================================================
            # [2.7.6] Seleção do Cenário Base
            # =================================================
            if (
                isinstance(simulations, dict)
                and "🟡 Base" in simulations
            ):

                narrative_simulation = (
                    simulations["🟡 Base"]
                )

            elif isinstance(simulations, dict):

                narrative_simulation = next(
                    iter(simulations.values())
                )

            else:

                narrative_simulation = simulations

            # =================================================
            # [2.7.7] Sanitização das Simulações
            # =================================================
            safe_simulations = {

                "strategy_name":
                    narrative_simulation.get(
                        "strategy_name",
                        "-"
                    ),

                "strategy_scope":
                    narrative_simulation.get(
                        "strategy_scope",
                        "-"
                    ),

                "target_reduction_pct":
                    narrative_simulation.get(
                        "target_reduction_pct",
                        0
                    ),

                "operational_complexity":
                    narrative_simulation.get(
                        "operational_complexity",
                        "Média"
                    ),

                "confidence_level":
                    narrative_simulation.get(
                        "confidence_level",
                        "Média"
                    ),

                "revenue_saved":
                    self.safe_float(
                        narrative_simulation.get(
                            "revenue_saved"
                        ),
                        0
                    ),

                "campaign_cost":
                    self.safe_float(
                        narrative_simulation.get(
                            "campaign_cost"
                        ),
                        0
                    ),

                "net_gain":
                    self.safe_float(
                        narrative_simulation.get(
                            "net_gain"
                        ),
                        0
                    ),

                "roi":
                    self.safe_float(
                        narrative_simulation.get(
                            "roi"
                        ),
                        0
                    )
            }

            # =================================================
            # [2.7.8] Construção dos Fatos Executivos
            # =================================================
            executive_facts = (
                self.build_executive_facts(
                    safe_kpis,
                    top_segments_clean,
                    safe_simulations
                )
            )

            # =====================================================
            # [2.7.9] System Prompt
            # Prompt de sistema para IA
            # =====================================================
            system_prompt = """
            Você é um Chief Revenue Officer (CRO)
            especializado em:

            - análise de churn
            - retenção de clientes
            - inteligência de receita
            - comunicação executiva corporativa

            Você escreve relatórios executivos:

            - quantitativos
            - objetivos
            - profissionais
            - sem exageros
            - sem inferência causal não comprovada

            REGRAS CRÍTICAS:

            - NÃO inventar explicações causais
            - NÃO assumir problemas operacionais
            - NÃO criar contexto não presente nos dados
            - NÃO repetir números excessivamente
            - NÃO transformar JSON em texto literal
            - NÃO escrever como dashboard
            - NÃO usar linguagem corporativa vazia
            - Sempre contextualizar ROI e impacto financeiro
            usando premissas operacionais realistas
            - Não prescrever ações como garantidas
            - Utilizar linguagem probabilística e analítica
            - Evitar termos absolutos como:
                "é fundamental", "garantirá", "irá reduzir"
            - Diferenciar claramente:
                potencial financeiro teórico
                versus cenários operacionais simulados
            - Sempre deixar claro que simulações
            representam cenários estimados
            e não resultados garantidos
            - Não repetir integralmente métricas
            já apresentadas em tabelas anteriores
            - Evitar frases genéricas como:
            "ganhos líquidos significativos",
            "impacto positivo relevante",
            "maximizar receita"

            - Preferir linguagem quantitativa e objetiva
            - Priorizar interpretação executiva
            sobre repetição numérica
            - Evitar linguagem excessivamente otimista
            - Evitar termos como:
                    "crucial",
                    "fundamental",
                    "essencial"

            OBJETIVO:

            Transformar dados analíticos
            em narrativa executiva clara.

            Escreva em português do Brasil.
            """

            # =====================================================
            # [2.7.10] User Prompt
            # Prompt com dados analíticos
            # =====================================================
            user_prompt = f"""
            KPIs EXECUTIVOS:

            {json.dumps(safe_kpis, indent=2, ensure_ascii=False)}


            FATOS EXECUTIVOS:

            {json.dumps(executive_facts, indent=2, ensure_ascii=False)}


            PRINCIPAIS SEGMENTOS EXECUTIVOS:

            {json.dumps(segments_list[:3], indent=2, ensure_ascii=False)}


            RESULTADO DAS SIMULAÇÕES:

            {json.dumps(safe_simulations, indent=2, ensure_ascii=False)}


            PRINCIPAIS RECOMENDAÇÕES:

            {json.dumps(recs_clean[:4], indent=2, ensure_ascii=False)}


            INSTRUÇÕES:

            1. Consolidar insights semelhantes

            2. Priorizar impacto financeiro

            3. Produzir narrativa objetiva

            4. Evitar repetição de segmentos

            5. Utilizar linguagem executiva realista

            6. Não criar causalidade não comprovada

            7. Não mencionar nomes técnicos internos

            8. Não repetir métricas excessivamente

            9. Diferenciar potencial bruto estimado
                de cenários operacionais simulados

            10. Priorizar interpretação
                em vez de repetir tabelas

            ESTRUTURA OBRIGATÓRIA:

            ## Resumo Executivo

            Máximo 2 parágrafos.

            ## Principais Concentrações de Risco

            Bullets quantitativos.

            ## Ações Prioritárias

            Bullets executivos.

            ## Perspectiva de Impacto Financeiro

            Resumo financeiro direto.
            """

            # =================================================
            # [2.7.11] Chamada da API Groq
            # =================================================
            response = (
                self.client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=[

                        {
                            "role": "system",
                            "content": system_prompt
                        },

                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    temperature=0.1,
                    max_tokens=1200,
                    top_p=0.85
                )
            )

            narrative = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            # =================================================
            # [2.7.12] Pós-processamento da Narrativa
            # =================================================
            narrative = translate_segment(
                narrative
            )

            narrative = narrative.replace(
                "Nãossa",
                "Nossa"
            )

            narrative = narrative.replace(
                "Não entanto",
                "No entanto"
            )

            # =================================================
            # [2.7.13] Tratamento de Resposta Vazia
            # =================================================
            if not narrative:

                return (
                    "❌ Erro ao gerar narrativa: "
                    "resposta vazia da IA."
                )

            return narrative

        # =================================================
        # [2.7.14] Tratamento de Exceções
        # =================================================
        except Exception as e:

            return (
                "❌ Erro ao gerar narrativa executiva: "
                f"{str(e)}"
            )