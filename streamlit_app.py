# ============================================================
# ÍNDICE
# ============================================================
#
# [1] IMPORTS
#     [1.1] Streamlit
#     [1.2] Pandas
#     [1.3] DataLayer
#     [1.4] AnalyticsLayer
#     [1.5] DecisionLayer
#     [1.6] NarrativeLayer
#     [1.7] ReportGenerator
#     [1.8] PredictiveLayer
#     [1.9] Translation Helpers
#
# [2] CONFIGURAÇÃO DA PÁGINA
#
# [3] CACHE
#     [3.1] load_data()
#     [3.2] build_engine()
#
# [4] SESSION STATE
#
# [5] CLASSE ChurnEngine
#
#     [5.1] Inicialização da Classe
#         [5.1.1] __init__()
#         [5.1.1.1] Dados Brutos
#         [5.1.1.2] Modelo Preditivo
#         [5.1.1.3] Dataset com Score
#         [5.1.1.4] Análise
#         [5.1.1.5] KPIs
#         [5.1.1.6] Segmentos
#         [5.1.1.7] Priorização
#         [5.1.1.8] Segmentos Executivos
#         [5.1.1.9] Recomendações
#         [5.1.1.10] Debug
#         [5.1.1.11] Identificar Nomes Reais das Colunas
#         [5.1.1.12] Recomendações com ROI Positivo
#         [5.1.1.13] Segmentos Lucrativos
#         [5.1.1.14] Simulações Financeiras
#
# [6] SIDEBAR
#
# [7] CARREGAR ENGINE
#
# [8] ABAS
#
# [9] TAB 1 - Visão Geral & KPIs
#
# [10] TAB 2 - Risco por Segmento
#     [10.1] Formatação Executiva
#     [10.2] Tradução Visual
#
# [11] TAB 3 - Ações & Simulações
#     [11.1] Recomendações
#     [11.2] Tradução Visual
#     [11.3] Simulações
#     [11.3.1] Card Financeiro
#
# [12] TAB 4 - Explicabilidade da IA
#
# [13] TAB 5 - Narrativa Executiva
#
# [14] TAB 6 - Exportar PDF
#     [14.1] Visualização PDF
#
# [15] RODAPÉ
#
# ============================================================


# ============================================================
# [1] IMPORTS
# ============================================================

# ============================================================
# [1.1] Streamlit
# Interface web analítica
# ============================================================
import streamlit as st

# ============================================================
# [1.2] Pandas
# Manipulação tabular
# ============================================================
import pandas as pd

# ============================================================
# [1.2] streamlit PDF 
# Correção na Visualização do PDF na hospedagem Streamlit Cloud
# ============================================================
import streamlit.components.v1 as components

# ============================================================
# [1.3] DataLayer
# Camada de carregamento de dados
# ============================================================
from src.data_layer import DataLayer

# ============================================================
# [1.4] AnalyticsLayer
# Camada analítica
# ============================================================
from src.analytics_layer import AnalyticsLayer

# ============================================================
# [1.5] DecisionLayer
# Camada de decisão
# ============================================================
from src.decision_layer import DecisionLayer

# ============================================================
# [1.6] NarrativeLayer
# Geração narrativa com IA
# ============================================================
from src.narrative_layer import NarrativeLayer

# ============================================================
# [1.7] ReportGenerator
# Geração de relatórios PDF
# ============================================================
from src.report_generator import ReportGenerator

# ============================================================
# [1.8] PredictiveLayer
# Modelo preditivo de churn
# ============================================================
from src.predictive_layer import PredictiveLayer

# ============================================================
# [1.9] Translation Helpers
# Traduções e labels executivos
# ============================================================
from src.utils.translations import translate_columns
from src.utils.translations import (
    translate_key,
    translate_shap_feature,
    translate_segment
)


# ============================================================
# [2] CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="ChurnSight AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(
    "🛡️ ChurnSight - Inteligência de Retenção com IA"
)

st.markdown(
    "**Plataforma de Inteligência para Decisão e Retenção de Clientes**"
)


# ============================================================
# [3] CACHE
# ============================================================

# ============================================================
# [3.1] load_data()
# Cache de carregamento do dataset
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():

    return DataLayer().get_data()


# ============================================================
# [3.2] build_engine()
# Cache do motor analítico
# ============================================================
@st.cache_resource(show_spinner=False)
def build_engine(
    campaign_reduction: int,
    top_n: int
):

    return ChurnEngine(
        campaign_reduction,
        top_n
    )


# ============================================================
# [4] SESSION STATE
# ============================================================
if "narrative" not in st.session_state:

    st.session_state.narrative = None

if "pdf_path" not in st.session_state:

    st.session_state.pdf_path = None


# ============================================================
# [5] CLASSE ChurnEngine
# Motor central da plataforma analítica
# ============================================================
class ChurnEngine:

    # ========================================================
    # [5.1] Inicialização da Classe
    # ========================================================

    # ========================================================
    # [5.1.1] __init__()
    # Inicializa pipeline analítico completo
    # ========================================================
    def __init__(
        self,
        campaign_reduction: int,
        top_n: int
    ):

        # ====================================================
        # [5.1.1.1] Dados Brutos
        # ====================================================
        self.data = load_data()

        # ====================================================
        # [5.1.1.2] Modelo Preditivo
        # ====================================================
        self.predictive = PredictiveLayer(
            self.data
        )

        self.model_metrics = (
            self.predictive.train_model()
        )

        self.shap_importance = (
            self.predictive
            .generate_shap_explanations()
        )

        # ====================================================
        # [5.1.1.3] Dataset com Score
        # ====================================================
        self.scored_df = (
            self.predictive
            .get_scored_dataset()
        )

        # ====================================================
        # [5.1.1.4] Análise
        # ====================================================
        self.analytics = AnalyticsLayer(
            self.scored_df
        )

        # ====================================================
        # [5.1.1.5] KPIs
        # ====================================================
        self.kpis = (
            self.analytics.kpi_agent()
        )

        # ====================================================
        # [5.1.1.6] Segmentos
        # ====================================================
        self.segments = (
            self.analytics.segment_risk_agent()
        )

        # ====================================================
        # [5.1.1.7] Priorização
        # ====================================================
        self.prioritized = (
            self.analytics.prioritization_agent(
                self.segments
            )
        )

        # ====================================================
        # [5.1.1.8] Segmentos Executivos
        # ====================================================
        self.executive_segments = (
            self.prioritized.copy()
        )

        # ====================================================
        # [5.1.1.9] Recomendações
        # ====================================================
        self.recommendations = (
            DecisionLayer.recommend_actions(
                self.prioritized,
                self.kpis
            )
        )

        # ====================================================
        # [5.1.1.10] Debug
        # ====================================================
        # st.write("Recommendations columns:")
        # st.write(self.recommendations.columns)

        # st.write("Prioritized columns:")
        # st.write(self.prioritized.columns)

        # ====================================================
        # [5.1.1.11] Identificar Nomes Reais das Colunas
        # ====================================================
        roi_col = (
            "ROI Estimado (%)"
            if "ROI Estimado (%)" in self.recommendations.columns
            else "estimated_roi"
        )

        segment_col = (
            "Segmento Executivo"
            if "Segmento Executivo" in self.recommendations.columns
            else "segment"
        )

        # ====================================================
        # [5.1.1.12] Recomendações com ROI Positivo
        # ====================================================
        self.profitable_recommendations = (
            self.recommendations[
                self.recommendations[roi_col] > 0
            ].copy()
        )

        # ====================================================
        # [5.1.1.13] Segmentos Lucrativos
        # ====================================================
        profitable_segments = (
            self.profitable_recommendations[
                segment_col
            ].unique()
        )

        self.profitable_segments = (
            self.prioritized[
                self.prioritized[
                    "executive_segment"
                ].isin(profitable_segments)
            ].copy()
        )

        # ====================================================
        # [5.1.1.14] Simulações Financeiras
        # ====================================================
        self.targeted_simulations = (
            DecisionLayer.simulate_targeted_campaign(
                prioritized=self.prioritized,
                top_n=top_n,
                churn_reduction_pct=campaign_reduction,
                total_revenue_at_risk=(
                    self.kpis.get(
                        "revenue_at_risk_annual",
                        0
                    )
                )
            )
        )


# ============================================================
# [6] SIDEBAR
# ============================================================
st.sidebar.header(
    "⚙️ Parâmetros da Análise"
)

top_n = st.sidebar.slider(
    "Top segmentos considerados",
    min_value=3,
    max_value=15,
    value=8
)

campaign_reduction = st.sidebar.slider(
    "Eficiência esperada da campanha (%)",
    min_value=5,
    max_value=25,
    value=10
)

show_charts = st.sidebar.checkbox(
    "Exibir gráficos avançados",
    value=True
)

if st.sidebar.button(
    "🔄 Recarregar análise completa",
    type="primary"
):

    st.cache_data.clear()
    st.cache_resource.clear()

    st.session_state.narrative = None
    st.session_state.pdf_path = None

    st.rerun()


# ============================================================
# [7] CARREGAR ENGINE
# ============================================================
try:

    # ========================================================
    # Placeholder temporário
    # ========================================================
    loading_message = st.empty()

    loading_message.info(
        """
        ⏳ O motor analítico está inicializando.

        Em hospedagens gratuitas do Streamlit Cloud,
        o primeiro carregamento pode levar mais tempo
        devido às limitações de CPU e memória.

        O sistema está processando:
        • Modelo preditivo de churn
        • Explicabilidade SHAP
        • Simulações financeiras
        • Estruturas analíticas
        """
    )

    with st.spinner(
        "Inicializando pipeline analítico..."
    ):

        engine = build_engine(
            campaign_reduction,
            top_n
        )

    # ========================================================
    # Remove mensagem após carregamento
    # ========================================================
    loading_message.empty()

except Exception as e:

    st.error(
        f"❌ Erro ao inicializar o motor analítico: {e}"
    )

    st.stop()


# ============================================================
# [8] ABAS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "📈 Visão Geral & KPIs",
    "🔥 Risco por Segmento",
    "🎯 Ações & Simulações",
    "🧠 Explicabilidade da IA",
    "📝 Narrativa Executiva",
    "📄 Exportar PDF"
])


# ============================================================
# [9] TAB 1 - Visão Geral & KPIs
# ============================================================
with tab1:

    st.subheader(
        "KPIs Principais"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Clientes Totais",
        f"{engine.kpis.get('total_customers', 0):,}"
    )

    col2.metric(
        "Taxa de Churn",
        f"{engine.kpis.get('churn_rate', 0):.2f}%"
    )

    col3.metric(
        "ARPU",
        f"US${engine.kpis.get('arpu', 0):,.2f}"
    )

    col4.metric(
        "Receita em Risco",
        f"US${engine.kpis.get('revenue_at_risk_annual', 0):,.0f}"
    )

    st.caption(
        f"""
        Receita Histórica Perdida:
        US${engine.kpis.get('historical_revenue_lost', 0):,.0f}
        """
    )

    if show_charts:

        colA, colB = st.columns(2)

        with colA:

            st.plotly_chart(
                engine.analytics.create_churn_pie(),
                use_container_width=True
            )

        with colB:

            st.plotly_chart(
                engine.analytics.create_risk_bar(
                    engine.profitable_segments
                ),
                use_container_width=True
            )
            # ============================================================
# [10] TAB 2 - Risco por Segmento
# ============================================================
with tab2:

    st.subheader(
        f"Top Segmentos de Risco"
    )

    display_df = (
        engine.profitable_segments
        .head(top_n)
        .copy()
    )

    # ========================================================
    # [10.1] Formatação Executiva
    # ========================================================
    if "revenue_at_risk_annual" in display_df.columns:

        display_df[
            "revenue_at_risk_annual"
        ] = (

            display_df[
                "revenue_at_risk_annual"
            ]

            .apply(
                lambda x:
                f"US$ {x:,.0f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
        )

    if "churn_rate" in display_df.columns:

        display_df[
            "churn_rate"
        ] = (

            display_df[
                "churn_rate"
            ]

            .apply(
                lambda x: f"{x:.1f}%"
            )
        )

    # ========================================================
    # [10.2] Tradução Visual
    # ========================================================
    for col in display_df.columns:

        display_df[col] = display_df[col].apply(
            lambda x: translate_segment(x)
            if isinstance(x, str)
            else x
        )

    st.dataframe(
        translate_columns(display_df),
        use_container_width=True,
        hide_index=True
    )

    if show_charts:

        st.plotly_chart(
            engine.analytics.create_segment_heatmap(),
            use_container_width=True
        )


# ============================================================
# [11] TAB 3 - Ações & Simulações
# ============================================================
with tab3:

    # ========================================================
    # [11.1] Recomendações
    # ========================================================
    st.subheader(
        "Recomendações Prioritárias"
    )

    recommendations_display = (
        engine.profitable_recommendations[[
            "Prioridade",
            "Segmento Executivo",
            "Ação Recomendada",
            "ROI Estimado (%)"
        ]]
        .copy()
    )

    # ========================================================
    # [11.2] Tradução Visual
    # ========================================================
    for col in recommendations_display.columns:

        recommendations_display[col] = (
            recommendations_display[col].apply(
                lambda x: translate_segment(x)
                if isinstance(x, str)
                else x
            )
        )

    st.dataframe(
        recommendations_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        """
        Segmentos com ROI projetado negativo
        foram excluídos das recomendações
        prioritárias para otimização do
        investimento operacional.
        """
    )

    st.divider()

    # ========================================================
    # [11.3] Simulações
    # ========================================================
    st.subheader(
        "Simulações Financeiras"
    )

    st.caption(
        """
        As projeções financeiras abaixo representam
        cenários analíticos estimados com base
        em premissas operacionais simplificadas.

        O modelo preditivo estima probabilidade
        de churn, enquanto os impactos financeiros
        são derivados de simulações de retenção.
        """
    )

    simulations = (
        engine.targeted_simulations
    )

    if not simulations:

        st.warning(
            "Nenhuma simulação disponível."
        )

    else:

        cols = st.columns(len(simulations))

        for idx, (name, data) in enumerate(
            simulations.items()
        ):

            with cols[idx]:

                revenue_saved = float(
                    data.get(
                        "revenue_saved",
                        0
                    )
                )

                net_gain = float(
                    data.get(
                        "net_gain",
                        0
                    )
                )

                campaign_cost = float(
                    data.get(
                        "campaign_cost",
                        0
                    )
                )

                roi = float(
                    data.get(
                        "display_roi",
                        data.get(
                            "roi",
                            0
                        )
                    )
                )

                assumptions = data.get(
                    "assumptions",
                    {}
                )

                # ====================================================
                # [11.3.1] Card Financeiro
                # ====================================================
                with st.container(border=True):

                    st.markdown(
                        f"### {name}"
                    )

                    st.metric(
                        "Receita Protegida",
                        f"US${revenue_saved:,.0f}",
                        delta=(
                            f"Ganho Líquido: "
                            f"US${net_gain:,.0f}"
                        )
                    )

                    st.metric(
                        "ROI Projetado",
                        f"{roi:.1f}%"
                    )

                    st.caption(
                        f"""
                        Faixa estimada de confiança:
                        {data.get('roi_uncertainty_band', '± 0%')}
                        """
                    )

                    st.caption(
                        f"""
                        Custo da Campanha:
                        US${campaign_cost:,.0f}
                        """
                    )

                    st.progress(
                        min(
                            max(roi / 100, 0),
                            1.0
                        )
                    )

                    st.markdown(
                        "##### Premissas"
                    )

                    st.caption(
                        f"""
                        • {translate_key('expected_churn_reduction')}:
                        {float(assumptions.get('expected_churn_reduction', 0)):.0%}

                        • {translate_key('contact_rate')}:
                        {float(assumptions.get('contact_rate', 0)):.0%}

                        • {translate_key('engagement_rate')}:
                        {float(assumptions.get('engagement_rate', 0)):.0%}

                        • {translate_key('retention_efficiency')}:
                        {float(assumptions.get('retention_efficiency', 0)):.0%}

                        • {translate_key('campaign_cost_ratio')}:
                        {float(assumptions.get('campaign_cost_ratio', 0)):.0%}

                        • Horizonte:
                        12 meses
                        """
                    )

                    methodology = assumptions.get(
                        "assumptions_methodology",
                        ""
                    )

                    if methodology:

                        with st.expander(
                            "Metodologia Financeira"
                        ):

                            st.info(methodology)


# ============================================================
# [12] TAB 4 - Explicabilidade da IA
# ============================================================
with tab4:

    st.subheader(
        "Explicabilidade do Modelo de Churn"
    )

    st.markdown(
        """
        Principais variáveis estatisticamente associadas
        ao aumento da probabilidade de churn
        segundo o modelo preditivo XGBoost.

        As importâncias representam tendências
        globais observadas no dataset
        e não causalidade individual.
        """
    )

    shap_df = (
        engine.shap_importance
        .head(10)
        .copy()
    )

    shap_display = shap_df.copy()

    shap_display["feature"] = (
        shap_display["feature"]
        .apply(translate_shap_feature)
    )

    st.dataframe(
        translate_columns(shap_display),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Interpretação Executiva"
    )

    explanations = {

        "Tenure":
            "Clientes com menor tempo de relacionamento apresentam maior risco de churn.",

        "Monthlycharges":
            "Valores mensais elevados aumentam a sensibilidade ao cancelamento.",

        "Customer Lifetime Value":
            "Clientes de maior valor exigem estratégias preventivas de retenção.",

        "Contract":
            "Contratos de longo prazo reduzem o risco de churn.",

        "Internetservice":
            "O tipo de serviço de internet impacta diretamente a retenção."
    }

    shown = set()

    for feature in shap_df["feature"]:

        for key, text in explanations.items():

            if (
                key.lower()
                in str(feature).lower()
                and key not in shown
            ):

                st.info(text)

                shown.add(key)


# ============================================================
# [13] TAB 5 - Narrativa Executiva
# ============================================================
with tab5:

    st.subheader(
        "Narrativa Executiva"
    )

    st.caption(
        """
        A narrativa abaixo é gerada por IA
        utilizando exclusivamente os dados,
        métricas e simulações calculadas
        pelo motor analítico do sistema.
        """
    )

    if st.button(
        "🚀 Gerar Narrativa Executiva (Groq)",
        type="primary"
    ):

        try:

            with st.spinner(
                "Gerando análise executiva com IA..."
            ):

                simulation_payload = (
                    engine.targeted_simulations
                )

                narrative = (
                    NarrativeLayer().generate(
                        engine.kpis,
                        engine.profitable_segments.head(top_n),
                        engine.targeted_simulations,
                        engine.profitable_recommendations
                    )
                )

                if narrative.startswith("❌"):

                    st.error(narrative)

                else:

                    st.session_state.narrative = (
                        narrative
                    )

                    st.success(
                        "Narrativa gerada com sucesso!"
                    )

        except Exception as e:

            st.error(
                f"Erro ao gerar narrativa: {e}"
            )

    if st.session_state.narrative:

        st.markdown(
    st.session_state.narrative,
    unsafe_allow_html=False
)

    else:

        st.info(
            "Clique no botão acima para gerar a narrativa executiva."
        )


# ============================================================
# [14] TAB 6 - Exportar PDF
# ============================================================
with tab6:

    st.subheader(
        "Relatório Executivo em PDF"
    )

    if st.button(
        "📑 Gerar Relatório PDF Completo",
        type="primary"
    ):

        try:

            with st.spinner(
                "Gerando relatório PDF..."
            ):

                if st.session_state.narrative:

                    narrative_text = (
                        st.session_state.narrative
                    )

                else:

                    simulation_payload = (
                        engine.targeted_simulations
                    )

                    narrative_text = (
                        NarrativeLayer().generate(
                            engine.kpis,
                            engine.executive_segments.head(top_n),
                            simulation_payload,
                            engine.recommendations
                        )
                    )

                    st.session_state.narrative = (
                        narrative_text
                    )

                generator = ReportGenerator()

                churn_fig = (
                    engine.analytics.create_churn_pie()
                )

                risk_fig = (
                    engine.analytics.create_risk_bar(
                        engine.profitable_segments
                    )
                )

                pdf_path = (
                    generator.generate_report(

                        kpis=engine.kpis,

                        segments=engine.profitable_segments.head(
                            top_n
                        ),

                        recommendations=(
                            engine.profitable_recommendations
                        ),

                        simulations=(
                            engine.targeted_simulations
                        ),

                        narrative=narrative_text,

                        model_metrics=(
                            engine.model_metrics
                        ),

                        shap_importance=(
                            engine.shap_importance
                        ),

                        churn_fig=(
                            churn_fig
                        ),

                        risk_fig=(
                            risk_fig
                        ),

                        filename=(
                            "Relatorio_Executivo_ChurnGuard.pdf"
                        )
                    )
                )

                st.session_state.pdf_path = (
                    pdf_path
                )

                st.success(
                    "✅ Relatório gerado com sucesso!"
                )

                with open(pdf_path, "rb") as f:

                    pdf_bytes = f.read()

                    st.download_button(
                        label="⬇️ Baixar PDF Executivo",
                        data=pdf_bytes,
                        file_name=(
                            "Relatorio_Executivo_ChurnGuard.pdf"
                        ),
                        mime="application/pdf"
                    )

                # ====================================================
                # [14.1] Visualização PDF
                # ====================================================
                st.markdown(
                    "### Visualização do Relatório"
                )
                st.info(
                    """
                    Alguns navegadores com bloqueadores
                    de privacidade agressivos (como Brave)
                    podem impedir a visualização inline do PDF.

                    Caso a visualização não apareça,
                    utilize o botão de download acima.
                    """
                )
                import base64
                import streamlit.components.v1 as components

                base64_pdf = base64.b64encode(
                    pdf_bytes
                ).decode("utf-8")

                pdf_display = f"""
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="100%"
                    height="1000px"
                    type="application/pdf">
                </iframe>
                """

                components.html(
                    pdf_display,
                    height=1000,
                    scrolling=True
                )

        except Exception as e:

            st.error(
                f"❌ Erro ao gerar PDF: {e}"
            )


# ============================================================
# [15] RODAPÉ
# ============================================================
st.divider()

st.caption(
    "ChurnSight AI • Plataforma de Inteligência de Decisão • "
    "Groq + Analytics Preditivo"
)