COLUMN_TRANSLATIONS = {

    # =====================================================
    # DATASET ORIGINAL
    # =====================================================
    "customerID":
        "ID Cliente",

    "crm_retention_scenario":
        "Modelo CRM de Retenção",

    "Churn":
        "Churn",

    "MonthlyCharges":
        "Cobrança Mensal",

    "TotalCharges":
        "Cobrança Total",

    "Contract":
        "Contrato",

    "InternetService":
        "Serviço de Internet",

    "tenure":
        "Tempo de Cliente",

    "tenure_bucket":
        "Faixa de Tempo",

    # =====================================================
    # SEGMENTAÇÃO
    # =====================================================
    "executive_segment":
        "Segmento Executivo",

    "detailed_segment":
        "Segmento Detalhado",

    "priority":
        "Prioridade",

    "priority_score":
        "Score Prioridade",

    "rank":
        "Ranking",

    "size":
        "Clientes",

    # =====================================================
    # FINANCEIRO
    # =====================================================
    "arpu":
        "ARPU",

    "revenue_at_risk_annual":
        "Receita em Risco (Anual)",

    "historical_revenue_lost":
        "Receita Perdida Histórica",

    "revenue_contribution_pct":
        "Contribuição Receita (%)",

    "avg_tenure":
        "Tempo Médio",

    "churn_rate":
        "Taxa de Churn (%)",

    "churned_customers":
        "Clientes Perdidos",

    # =====================================================
    # CONTRATOS
    # =====================================================
    "dominant_contract":
        "Contrato Dominante",

    "dominant_internet":
        "Internet Dominante",

    # =====================================================
    # RECOMENDAÇÕES
    # =====================================================
    "Recommended Action":
        "Ação Recomendada",

    "Estimated ROI (%)":
        "ROI Estimado (%)",

    "Executive Segment":
        "Segmento Executivo",

    "Priority":
        "Prioridade",

    # =====================================================
    # KPIs EXECUTIVOS
    # =====================================================
    "total_customers":
        "Total de Clientes",

    "avg_probability":
        "Probabilidade Média de Churn",

    "high_risk_customers":
        "Clientes de Alto Risco",

    "auc":
        "ROC-AUC",

    # =====================================================
    # SIMULAÇÃO FINANCEIRA
    # =====================================================
    "target_segments":
        "Segmentos Prioritários",

    "expected_churn_reduction":
        "Redução Esperada de Churn",

    "contact_rate":
        "Taxa de Contato",

    "engagement_rate":
        "Taxa de Engajamento",

    "retention_efficiency":
        "Eficiência de Retenção",

    "retention_efficiency_note":
        "Referência Operacional",

    "campaign_cost_ratio":
        "Custo Operacional da Campanha",

    "campaign_cost_per_customer":
        "Custo por Cliente Impactado",

    "avg_retention_months":
        "Meses Médios de Retenção",

    "time_horizon_months":
        "Horizonte Financeiro",

    "financial_model":
        "Modelo Financeiro",

    "baseline_revenue_at_risk":
        "Receita Base em Risco",

    "strategy_name":
        "Estratégia",

    "strategy_scope":
        "Escopo Estratégico",

    "target_reduction_pct":
        "Meta de Redução (%)",

    "operational_complexity":
        "Complexidade Operacional",

    "confidence_level":
        "Nível de Confiança",

    "revenue_saved":
        "Receita Protegida",

    "campaign_cost":
        "Custo da Campanha",

    "net_gain":
        "Ganho Líquido",

    "roi":
        "ROI",

    "display_roi":
        "ROI Projetado",

    "Feature":
        "Variável",

    "importance":
        "Importância",
    "contact_rate":
    "Taxa de Contato",

    "engagement_rate":
        "Taxa de Engajamento",

    "time_horizon_months":
        "Horizonte Financeiro (Meses)",

    "campaign_cost_per_customer":
        "Custo por Cliente Impactado",

    "avg_retention_months":
        "Meses Médios de Retenção",
}
SHAP_FEATURE_TRANSLATIONS = {

    "tenure": "Tempo de Relacionamento",

    "MonthlyCharges": "Cobrança Mensal",

    "TotalCharges": "Cobrança Total",

    "SeniorCitizen": "Cliente Idoso",

    "Partner": "Possui Parceiro(a)",

    "Dependents": "Possui Dependentes",

    "PhoneService": "Serviço Telefônico",

    "MultipleLines": "Múltiplas Linhas",

    "InternetService": "Serviço de Internet",

    "OnlineSecurity": "Segurança Online",

    "OnlineBackup": "Backup Online",

    "DeviceProtection": "Proteção de Dispositivo",

    "TechSupport": "Suporte Técnico",

    "StreamingTV": "Streaming TV",

    "StreamingMovies": "Streaming Filmes",

    "Contract": "Tipo de Contrato",

    "PaperlessBilling": "Faturamento Digital",

    "PaymentMethod": "Método de Pagamento",

    "MonthlyCharges": "Cobrança Mensal",

    "TotalCharges": "Cobrança Total"
}


def translate_shap_feature(feature):

    translations = {

        "tenure":
            "Tempo de Permanência",

        "MonthlyCharges":
            "Cobrança Mensal",

        "TotalCharges":
            "Cobrança Total",

        "customer lifetime value":
            "Valor do Cliente",

        "is new customer":
            "Cliente Novo",

        "high value customer":
            "Cliente Alto Valor",

        "InternetService Fiber optic":
            "Internet Fibra Óptica",

        "InternetService No":
            "Sem Internet",

        "Contract Two year":
            "Contrato de 2 Anos",

        "Contract One year":
            "Contrato de 1 Ano",

        "PaymentMethod Electronic check":
            "Cheque Eletrônico",

        "TechSupport No":
            "Sem Suporte Técnico",

        "OnlineBackup No":
            "Sem Backup Online",

        "OnlineSecurity No":
            "Sem Segurança Online",

        "StreamingTV Yes":
            "Streaming TV",

        "StreamingMovies Yes":
            "Streaming Filmes",

        "PaperlessBilling Yes":
            "Fatura Digital",

        "SeniorCitizen":
            "Idoso",

        "Partner Yes":
            "Possui Parceiro",

        "Dependents Yes":
            "Possui Dependentes",

        "MultipleLines Yes":
            "Múltiplas Linhas",

        "PhoneService Yes":
            "Serviço Telefônico",
        "month to month":
            "Contrato Mensal",

        "Tempo de Permanência":
            "Tempo de Permanência",

        "avg monthly revenue":
            "Receita Média Mensal",

        "Cobrança Total":
            "Cobrança Total",

        "Contrato de 2 Anos":
            "Contrato de 2 Anos",

        "Cobrança Mensal":
            "Cobrança Mensal",

        "has fiber":
            "Possui Fibra Óptica",

        "Valor do Cliente":
            "Valor do Cliente",

        "electronic check":
            "Cheque Eletrônico",

        "TechSupport Yes":
            "Possui Suporte Técnico"
    
    }

    return translations.get(feature, feature)
# =====================================================
# SEGMENTOS 
# =====================================================
def translate_segment(text):

    if not isinstance(text, str):

        return text

    executive_translations = {

        "High Risk Fiber Customers":
            "Clientes Fibra de Alto Risco",

        "High Risk DSL Customers":
            "Clientes DSL de Alto Risco",

        "Mid-Term Fiber Customers":
            "Clientes Fibra de Médio Prazo",

        "Mid-Term DSL Customers":
            "Clientes DSL de Médio Prazo",

        "Loyal Fiber Customers":
            "Clientes Fibra Fiéis",

        "Loyal DSL Customers":
            "Clientes DSL Fiéis",

        "Month-to-month | No":
            "Mensal | Não",

        "Two year | No":
            "2 Anos | Não",

        "One year | No":
            "1 Ano | Não",
    }

    translations = {

        "Month-to-month": "Mensal",
        "One year": "1 Ano",
        "Two year": "2 Anos",

        "Fiber optic": "Fibra Óptica",
        "DSL": "DSL",

        "Electronic check": "Cheque Eletrônico",
        "Mailed check": "Cheque por Correio",
        "Bank transfer": "Transferência Bancária",
        "Credit card": "Cartão de Crédito",

        "No internet service": "Sem Internet",
        "No phone service": "Sem Telefone",

        "Yes": "Sim",
        "No": "Não",
        "Critical": "Crítica",
        "High": "Alta",
        "Medium": "Média",
    }

    result = str(text)

    for eng, pt in executive_translations.items():

        result = result.replace(eng, pt)

    for eng, pt in translations.items():

        result = result.replace(eng, pt)

    return result

    
# =====================================================
# TRADUZ COLUNAS DE DATAFRAME
# =====================================================
def translate_columns(df):

    return df.rename(
        columns=COLUMN_TRANSLATIONS
    )


# =====================================================
# TRADUZ CHAVE INDIVIDUAL
# =====================================================
def translate_key(key):

    return COLUMN_TRANSLATIONS.get(
        key,
        str(key)
        .replace("_", " ")
        .title()
    )