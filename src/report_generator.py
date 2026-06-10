import os
import pandas as pd
import numpy as np
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import KeepTogether
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib import colors

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image
)
from src.utils.translations import (
    COLUMN_TRANSLATIONS,
    translate_key,
    translate_segment,
    translate_shap_feature
)

class ReportGenerator:

    def __init__(self):

        self.styles = (
            getSampleStyleSheet()
        )

        # ==================================================
        # TITLE
        # ==================================================
        self.title_style = ParagraphStyle(

            'CustomTitle',

            parent=self.styles['Heading1'],

            fontSize=20,

            leading=24,

            textColor=colors.HexColor(
                "#111827"
            ),

            spaceAfter=18
        )

        # ==================================================
        # SECTION
        # ==================================================
        self.section_style = ParagraphStyle(

            'CustomSection',

            parent=self.styles['Heading2'],

            fontSize=14,

            leading=18,

            textColor=colors.HexColor(
                "#1f2937"
            ),

            spaceAfter=10
        )

        # ==================================================
        # NORMAL
        # ==================================================
        self.normal_style = ParagraphStyle(

            'CustomNormal',

            parent=self.styles['BodyText'],

            fontSize=10,

            leading=15,

            textColor=colors.HexColor(
                "#374151"
            )
        )

        # ==================================================
        # SMALL
        # ==================================================
        self.small_style = ParagraphStyle(

            'SmallText',

            parent=self.styles['BodyText'],

            fontSize=8,

            leading=11,

            textColor=colors.HexColor(
                "#111827"
            )
        )

    # ======================================================
    # SAFE TEXT
    # ======================================================
    def safe_text(self, value):

        if value is None:
            return "-"

        try:

            if pd.isna(value):
                return "-"

        except:
            pass

        if isinstance(
            value,
            (
                np.float64,
                np.float32,
                float
            )
        ):

            return f"{value:,.2f}"

        if isinstance(
            value,
            (
                np.int64,
                np.int32,
                int
            )
        ):

            return f"{value:,}"

        return str(value)

    # ======================================================
    # SAFE FLOAT
    # ======================================================
    def safe_float(
        self,
        value,
        default=0.0
    ):

        try:

            if value is None:
                return default

            if pd.isna(value):
                return default

            return float(value)

        except:
            return default

    # ======================================================
    # KPI TABLE
    # ======================================================
    def simple_kpi_table(
        self,
        data_dict
    ):

        rows = [["Métrica", "Valor"]]

        for k, v in data_dict.items():

            pretty_key = translate_key(k)

            rows.append([

                pretty_key,

                self.safe_text(v)
            ])

        table = Table(
            rows,
            colWidths=[240, 220]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND',
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")),

            ('TEXTCOLOR',
                (0, 0),
                (-1, 0),
                colors.white),

            ('GRID',
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey),

            ('FONTNAME',
                (0, 0),
                (-1, 0),
                'Helvetica-Bold'),

            ('BACKGROUND',
                (0, 1),
                (-1, -1),
                colors.whitesmoke),

            ('BOTTOMPADDING',
                (0, 0),
                (-1, 0),
                10),
        ]))

        return table

    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================
    def model_performance_table(
        self,
        metrics: dict
    ):

        metrics = metrics or {}

        rows = [

            ["Métrica", "Valor"],

            [
                "Tipo de Modelo",
                "XGBoost Classifier"
            ],

            [
                "ROC-AUC",
                f"{self.safe_float(metrics.get('auc', 0)):.4f}"
            ],

            [
                "Probabilidade Média de Churn",
                f"{self.safe_float(metrics.get('avg_probability', 0)):.2%}"
            ],

            [
                "Clientes de Alto Risco",
                f"{int(self.safe_float(metrics.get('high_risk_customers', 0))):,}"
            ]
        ]

        table = Table(
            rows,
            colWidths=[240, 220]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND',
             (0, 0),
             (-1, 0),
             colors.HexColor("#1f2937")),

            ('TEXTCOLOR',
             (0, 0),
             (-1, 0),
             colors.white),

            ('GRID',
             (0, 0),
             (-1, -1),
             0.5,
             colors.grey),

            ('FONTNAME',
             (0, 0),
             (-1, 0),
             'Helvetica-Bold'),

            ('BACKGROUND',
             (0, 1),
             (-1, -1),
             colors.whitesmoke),
        ]))

        return table

    # ======================================================
    # EXECUTIVE SEGMENTS TABLE
    # ======================================================
    def segments_table(
        self,
        df: pd.DataFrame
    ):

        if df is None or df.empty:

            return Paragraph(
                "Nenhum dado de segmento disponível.",
                self.normal_style
            )

        top = df.head(10).copy()

        rows = [[

            "Prioridade",
            "Segmento Executivo",
            "Clientes",
            "Churn %",
            "Receita em Risco"
        ]]

        for _, row in top.iterrows():

            rows.append([

                self.safe_text(
                    translate_segment(
                        row.get(
                            "priority",
                            "-"
                        )
                    )
                ),

                self.safe_text(
                    translate_segment(
                        row.get(
                            "executive_segment",
                            "-"
                        )
                    )
                ),

                self.safe_text(
                    row.get(
                        "size",
                        0
                    )
                ),

                f"{self.safe_float(row.get('churn_rate', 0)):.1f}%",

                f"US${self.safe_float(row.get('revenue_at_risk_annual', 0)):,.0f}"
            ])

        table = Table(
            rows,
            colWidths=[70, 220, 70, 70, 90]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND',
             (0, 0),
             (-1, 0),
             colors.HexColor("#991b1b")),

            ('TEXTCOLOR',
             (0, 0),
             (-1, 0),
             colors.white),

            ('GRID',
             (0, 0),
             (-1, -1),
             0.5,
             colors.grey),

            ('FONTNAME',
             (0, 0),
             (-1, 0),
             'Helvetica-Bold'),

            ('BACKGROUND',
             (0, 1),
             (-1, -1),
             colors.whitesmoke),
        ]))

        return table

    # ======================================================
    # RECOMMENDATIONS TABLE
    # ======================================================
    def recommendations_table(
        self,
        df: pd.DataFrame
    ):

        if df is None or df.empty:

            return Paragraph(
                "Nenhuma recomendação disponível.",
                self.normal_style
            )

        rows = [[

            "Prioridade",
            "Segmento Executivo",
            "Ação Recomendada",
            "ROI"
        ]]

        filtered_df = (
            df[
                df["ROI Estimado (%)"] > 0
            ]
            .head(8)
        )

        for _, row in filtered_df.iterrows():

            priority = translate_segment(
                row.get(
                    "Prioridade",
                    "-"
                )
            )

            segment = translate_segment(
                row.get(
                    "Segmento Executivo",
                    "-"
                )
            )

            action = row.get(
                "Ação Recomendada",
                "-"
            )

            roi = row.get(
                "ROI Estimado (%)",
                0
            )

            rows.append([

            Paragraph(
                self.safe_text(priority),
                self.small_style
            ),

            Paragraph(
                self.safe_text(segment),
                self.small_style
            ),

            Paragraph(
                self.safe_text(action),
                self.small_style
            ),

            Paragraph(
                f"{float(roi):.1f}%",
                self.small_style
            )
            ])

        table = Table(
            rows,
            colWidths=[60, 140, 220, 50]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND',
            (0, 0),
            (-1, 0),
            colors.HexColor("#065f46")),

            ('TEXTCOLOR',
            (0, 0),
            (-1, 0),
            colors.white),

            ('GRID',
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey),

            ('FONTNAME',
            (0, 0),
            (-1, 0),
            'Helvetica-Bold'),

            ('BACKGROUND',
            (0, 1),
            (-1, -1),
            colors.whitesmoke),

            ('FONTSIZE',
            (0, 0),
            (-1, -1),
            8),

            ('VALIGN',
            (0, 0),
            (-1, -1),
            'MIDDLE'),

            ('BOTTOMPADDING',
            (0, 0),
            (-1, -1),
            6),
        ]))

        return table

    # ======================================================
    # SHAP TABLE
    # ======================================================
    def shap_table(
        self,
        shap_df: pd.DataFrame
    ):

        if shap_df is None or shap_df.empty:

            return Paragraph(
                "Nenhum dado de explicabilidade disponível.",
                self.normal_style
            )

        top = shap_df.head(10)

        rows = [["feature", "Importância"]]

        for _, row in top.iterrows():

            rows.append([

                self.safe_text(
                    translate_shap_feature(
                        row["feature"]
                    )
                ),

                f"{self.safe_float(row['importance']):.4f}"
            ])

        table = Table(
            rows,
            colWidths=[300, 140]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND',
                (0, 0),
                (-1, 0),
                colors.HexColor("#7c3aed")),

            ('TEXTCOLOR',
                (0, 0),
                (-1, 0),
                colors.white),

            ('GRID',
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey),

            ('FONTNAME',
                (0, 0),
                (-1, 0),
                'Helvetica-Bold'),

            ('BACKGROUND',
                (0, 1),
                (-1, -1),
                colors.whitesmoke),
        ]))

        return table

    # ======================================================
    # SIMULATION TABLE
    # ======================================================
    def simulations_to_table(
        self,
        simulations
    ):

        if (
            not simulations
            or not isinstance(simulations, dict)
        ):

            return Paragraph(
                "Nenhum dado de simulação disponível.",
                self.normal_style
            )

        rows = [[

            "Cenário",
            "Escopo",
            "Receita Protegida",
            "Custo",
            "ROI",
            "Faixa ROI",
            "Confiança"
        ]]

        row_colors = []

        for scenario_name, values in simulations.items():

            if not isinstance(values, dict):
                continue

            revenue_saved = float(
                values.get(
                    "revenue_saved",
                    0
                )
            )

            campaign_cost = float(
                values.get(
                    "campaign_cost",
                    0
                )
            )

            roi = float(
                values.get(
                    "roi",
                    0
                )
            )

            if roi <= 0:
                continue

            strategy_scope = translate_segment(
                values.get(
                    "strategy_scope",
                    "-"
                )
            )

            confidence = values.get(
                "confidence_level",
                "-"
            )
            roi_range = values.get(
                "roi_range",
                "-"
            )
            # ==========================================
            # LIMITA TAMANHO DO TEXTO
            # ==========================================
            strategy_scope = str(strategy_scope)[:60]

            rows.append([

                    Paragraph(
                        str(scenario_name),
                        self.small_style
                    ),

                    Paragraph(
                        strategy_scope,
                        self.small_style
                    ),

                    Paragraph(
                        f"US${revenue_saved:,.0f}",
                        self.small_style
                    ),

                    Paragraph(
                        f"US${campaign_cost:,.0f}",
                        self.small_style
                    ),

                    Paragraph(
                        f"{roi:.1f}%",
                        self.small_style
                    ),

                    Paragraph(
                        str(roi_range),
                        self.small_style
                    ),

                    Paragraph(
                        str(confidence),
                        self.small_style
                    )
                ])

            # ==========================================
            # COR POR CENÁRIO
            # ==========================================
            if "Conservador" in scenario_name:

                row_colors.append(
                    colors.HexColor("#d1fae5")
                )

            elif "Base" in scenario_name:

                row_colors.append(
                    colors.HexColor("#fef3c7")
                )

            else:

                row_colors.append(
                    colors.HexColor("#fee2e2")
                )

        table = Table(

            rows,

            colWidths=[60, 120, 75, 65, 45, 70, 45],
            splitByRow=1,
            repeatRows=1
        )

        style_commands = [

            # HEADER
            ('BACKGROUND',
            (0, 0),
            (-1, 0),
            colors.HexColor("#111827")),

            ('TEXTCOLOR',
            (0, 0),
            (-1, 0),
            colors.white),

            ('FONTNAME',
            (0, 0),
            (-1, 0),
            'Helvetica-Bold'),

            ('BOTTOMPADDING',
            (0, 0),
            (-1, 0),
            10),

            # GRID
            ('GRID',
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey),

            # ALIGNMENT
            ('VALIGN',
            (0, 0),
            (-1, -1),
            'MIDDLE'),

            # FONT
            ('FONTSIZE',
            (0, 0),
            (-1, -1),
            8),

            # WRAP
            ('WORDWRAP',
            (0, 0),
            (-1, -1),
            True),

            # PADDING
            ('TOPPADDING',
            (0, 0),
            (-1, -1),
            6),

            ('BOTTOMPADDING',
            (0, 0),
            (-1, -1),
            6),

            ('LEFTPADDING',
            (0, 0),
            (-1, -1),
            4),

            ('RIGHTPADDING',
            (0, 0),
            (-1, -1),
            4),
        ]

        # ==========================================
        # APLICA COR DAS LINHAS
        # ==========================================
        for idx, color in enumerate(row_colors, start=1):

            style_commands.append(

                ('BACKGROUND',
                (0, idx),
                (-1, idx),
                color)
            )

        table.setStyle(
            TableStyle(style_commands)
        )

        return table
    
        # ======================================================
    # FINANCIAL ASSUMPTIONS TABLE
    # ======================================================
    def assumptions_table(
        self,
        simulations
    ):

        if (
            not simulations
            or not isinstance(simulations, dict)
        ):

            return [
                Paragraph(
                    "Nenhuma premissa disponível.",
                    self.normal_style
                )
            ]

        elements = []

        for scenario_name, values in simulations.items():

            if not isinstance(values, dict):
                continue

            assumptions = values.get(
                "assumptions",
                {}
            )

            if not assumptions:
                continue

            # ==========================================
            # TITULO DO CENÁRIO
            # ==========================================
            elements.append(

                Paragraph(
                    f"<b>{scenario_name}</b>",
                    self.normal_style
                )
            )

            elements.append(
                Spacer(1, 6)
            )

            rows = [[
                "Premissa",
                "Valor"
            ]]

            for key, value in assumptions.items():

                if key == "assumptions_methodology":
                    continue

                pretty_key = translate_key(key)

                if isinstance(
                    value,
                    (float, int)
                ):

                    percent_fields = [

                        "expected_churn_reduction",
                        "contact_rate",
                        "engagement_rate",
                        "retention_efficiency",
                        "campaign_cost_ratio"
                    ]

                    currency_fields = [

                        "campaign_cost_per_customer",
                        "baseline_revenue_at_risk"
                    ]

                    if key in percent_fields:

                        display = f"{float(value):.0%}"

                    elif key in currency_fields:

                        display = f"US${float(value):,.2f}"

                    else:

                        display = f"{float(value):,.2f}"

                else:

                    display = translate_key(
                        str(value)
                    )

                rows.append([
                    pretty_key,
                    display
                ])

            table = Table(
                rows,
                colWidths=[260, 180]
            )

            # ==========================================
            # COR POR CENÁRIO
            # ==========================================
            if "Conservador" in scenario_name:

                header_color = colors.HexColor("#065f46")

            elif "Base" in scenario_name:

                header_color = colors.HexColor("#92400e")

            else:

                header_color = colors.HexColor("#991b1b")

            table.setStyle(TableStyle([

                ('BACKGROUND',
                 (0, 0),
                 (-1, 0),
                 header_color),

                ('TEXTCOLOR',
                 (0, 0),
                 (-1, 0),
                 colors.white),

                ('GRID',
                 (0, 0),
                 (-1, -1),
                 0.5,
                 colors.grey),

                ('FONTNAME',
                 (0, 0),
                 (-1, 0),
                 'Helvetica-Bold'),

                ('BACKGROUND',
                 (0, 1),
                 (-1, -1),
                 colors.whitesmoke),

                ('BOTTOMPADDING',
                 (0, 0),
                 (-1, 0),
                 10),
            ]))

            elements.append(table)

            elements.append(
                Spacer(1, 16)
            )

        return elements
    
    # ======================================================
    # NARRATIVE
    # ======================================================
    def narrative_to_paragraphs(
        self,
        text
    ):

        elements = []

        if not text:

            elements.append(

                Paragraph(
                    "Nenhuma narrativa executiva disponível.",
                    self.normal_style
                )
            )

            return elements

        paragraphs = (
            str(text)
            .split("\n")
        )

        for p in paragraphs:

            p = p.strip()

            if not p:
                continue

            if p.startswith("##"):

                title = (
                    p.replace("##", "")
                    .strip()
                )

                elements.append(

                    Paragraph(
                        title,
                        self.section_style
                    )
                )

            elif p.startswith("-"):

                bullet = (
                    p.replace("-", "•", 1)
                )

                elements.append(

                    Paragraph(
                        bullet,
                        self.normal_style
                    )
                )

            else:

                elements.append(

                    Paragraph(
                        p,
                        self.normal_style
                    )
                )

            elements.append(
                Spacer(1, 6)
            )

        return elements

    # ======================================================
    # GENERATE REPORT
    # ======================================================
    def generate_report(
        self,
        kpis,
        segments,
        recommendations,
        simulations,
        narrative,
        model_metrics=None,
        shap_importance=None,
        churn_fig=None,
        risk_fig=None,
        filename="relatorio_executivo.pdf"
    ):

        output_dir = "reports"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        pdf_path = os.path.join(
            output_dir,
            filename
        )

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )

        story = []

        # ==================================================
        # TITLE
        # ==================================================
        story.append(

            Paragraph(
                "Relatório Executivo de Inteligência de Churn",
                self.title_style
            )
        )

        story.append(
            Spacer(1, 14)
        )

        # ==================================================
        # KPIS
        # ==================================================
        story.append(

            Paragraph(
                "1. KPIs Executivos",
                self.section_style
            )
        )

        story.append(
            self.simple_kpi_table(kpis)
        )

        story.append(
            Spacer(1, 16)
        )

        # ==================================================
        # CHURN PIE CHART
        # ==================================================
        if churn_fig is not None:

            try:

                churn_path = tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ).name

                churn_fig.write_image(
                    churn_path,
                    width=800,
                    height=450
                )

                story.append(

                    Paragraph(
                        "Distribuição Geral de Churn",
                        self.section_style
                    )
                )

                story.append(

                    Image(
                        churn_path,
                        width=420,
                        height=240
                    )
                )

                story.append(
                    Spacer(1, 20)
                )

            except Exception as e:

                story.append(

                    Paragraph(
                        f"Erro ao gerar gráfico de churn: {e}",
                        self.normal_style
                    )
                )

        # ==================================================
        # MODEL PERFORMANCE
        # ==================================================
        if model_metrics:

            story.append(

                Paragraph(
                    "2. Performance do Modelo Preditivo",
                    self.section_style
                )
            )

            story.append(

                self.model_performance_table(
                    model_metrics
                )
            )

            story.append(
                Spacer(1, 20)
            )

        # ==================================================
        # EXECUTIVE SEGMENTS
        # ==================================================
        story.append(

            Paragraph(
                "3. Segmentos Executivos de Risco",
                self.section_style
            )
        )

        story.append(
            self.segments_table(segments)
        )

        story.append(
            Spacer(1, 16)
        )

        # ==================================================
        # RISK BAR CHART
        # ==================================================
        if risk_fig is not None:

            try:

                risk_path = tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ).name

                # ==================================================
                # AJUSTE EXCLUSIVO PARA PDF
                # ==================================================
                risk_fig.update_layout(

                    height=850,
                    title=None,
                    margin=dict(
                        l=340,
                        r=40,
                        t=60,
                        b=40
                    ),

                    yaxis=dict(
                        title=None,
                        tickfont=dict(size=22),
                        automargin=True
                    )
                )

                risk_fig.write_image(
                    risk_path,
                    width=1400,
                    height=850
                )

                story.append(

                    Paragraph(
                        "Receita em Risco por Segmento",
                        self.section_style
                    )
                )

                story.append(

                    Image(
                        risk_path,
                        width=450,
                        height=360
                    )
                )

                story.append(
                    Spacer(1, 20)
                )

            except Exception as e:

                story.append(

                    Paragraph(
                        f"Erro ao gerar gráfico de risco: {e}",
                        self.normal_style
                    )
                )

        # ==================================================
        # RECOMMENDATIONS
        # ==================================================
        story.append(

            Paragraph(
                "4. Prioridades de Retenção",
                self.section_style
            )
        )

        story.append(

            self.recommendations_table(
                recommendations
            )
        )
        story.append(
            Spacer(1, 8)
        )

        story.append(
            Paragraph(
                (
                    "Segmentos com ROI projetado negativo "
                    "foram excluídos das recomendações "
                    "prioritárias para otimização do "
                    "investimento operacional."
                ),
                self.small_style
            )
        )
        story.append(
            Spacer(1, 20)
        )

        # ==================================================
        # SHAP
        # ==================================================
        if (
            shap_importance is not None
            and not shap_importance.empty
        ):

            story.append(

                Paragraph(
                    "5. Principais Drivers de Churn",
                    self.section_style
                )
            )

            story.append(

                self.shap_table(
                    shap_importance
                )
            )

            story.append(
                Spacer(1, 20)
            )

        # ==================================================
        # SIMULATION
        # ==================================================
        story.append(

            Paragraph(
                "6. Simulação de Impacto Financeiro",
                self.section_style
            )
        )

        story.append(

            Paragraph(
                (
                    "As estimativas financeiras abaixo "
                    "são baseadas em premissas operacionais "
                    "conservadoras de campanhas de retenção."
                ),
                self.normal_style
            )
        )

        story.append(
            Spacer(1, 10)
        )

        story.append(

            KeepTogether(

                self.simulations_to_table(
                    simulations
                )
            )
        )

        story.append(
            Spacer(1, 16)
        )

        # ==================================================
        # ASSUMPTIONS
        # ==================================================
        story.append(

            Paragraph(
                "6.1 Premissas Financeiras",
                self.section_style
            )
        )

        story.append(

            Paragraph(
                (
                    "O ROI projetado considera hipóteses "
                    "simplificadas de conversão e custo "
                    "operacional. Os valores representam "
                    "cenários analíticos estimados e não "
                    "garantias financeiras."
                ),
                self.normal_style
            )
        )

        story.append(
            Spacer(1, 10)
        )

        # ==================================================
        # ASSUMPTIONS METHODOLOGY
        # ==================================================
        try:

            first_simulation = next(
                iter(simulations.values())
            )

            assumptions = first_simulation.get(
                "assumptions",
                {}
            )

            methodology = assumptions.get(
                "assumptions_methodology",
                ""
            )

            if methodology:

                story.append(

                    Paragraph(
                        methodology,
                        self.normal_style
                    )
                )

                story.append(
                    Spacer(1, 10)
                )

        except:
            pass

        story.extend(

            self.assumptions_table(
                simulations
            )
        )

        story.append(
            Spacer(1, 20)
        )
        # ==================================================
        # METODOLOGIA
        # ==================================================
        story.append(

            Paragraph(
                "7. Metodologia Financeira",
                self.section_style
            )
        )

        methodology_text = """
        O modelo preditivo deste relatório
        estima probabilidade de churn.

        Os impactos financeiros apresentados
        foram calculados através de simulações
        analíticas baseadas em premissas operacionais
        e não representam previsões financeiras garantidas.
        """

        story.append(

            Paragraph(
                methodology_text,
                self.normal_style
            )
        )

        story.append(
            Spacer(1, 20)
        )

        story.append(

            Paragraph(
                "8. Limitações Analíticas",
                self.section_style
            )
        )

        limitations = """
        As análises apresentadas dependem
        da qualidade e granularidade dos dados disponíveis.

        Mudanças operacionais, comportamento de mercado,
        ações competitivas e fatores externos
        não observados no dataset
        podem impactar os resultados reais.

        As probabilidades de churn representam
        tendências estatísticas
        e não comportamento individual garantido.
        """

        story.append(

            Paragraph(
                limitations,
                self.normal_style
            )
        )

        story.append(
            Spacer(1, 20)
        )

        # ==================================================
        # PAGE BREAK
        # ==================================================
        story.append(
            PageBreak()
        )

        # ==================================================
        # EXECUTIVE NARRATIVE
        # ==================================================
        story.append(

            Paragraph(
                "9. Narrativa Executiva",
                self.section_style
            )
        )

        story.extend(

            self.narrative_to_paragraphs(
                narrative
            )
        )

        # ==================================================
        # BUILD PDF
        # ==================================================
        doc.build(story)

        return pdf_path