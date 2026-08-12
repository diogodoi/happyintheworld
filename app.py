import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuração inicial da página
st.set_page_config(
    page_title="Dashboard - Felicidade Mundial",
    page_icon="🌍",
    layout="wide"
)

# Título e Descrição
st.title("🌍 Dashboard: Análise da Felicidade ao Redor do Mundo")
st.markdown("""
**MBA em IA e BIGDATA | CURSO 2 - CD, AM e DM**  
Este dashboard explora os dados do *World Happiness Report 2024*. Utilizamos métricas como PIB per capita, suporte social e expectativa de vida para entender o que impulsiona a felicidade (Life Ladder).

Desenvolvido por: **Diogo Godoi** \n
Codigo disponível: [GitHub](https://github.com/diogodoi/happyintheworld)

""")

# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ---
@st.cache_data
def load_data():
    # Lendo os datasets
    df1 = pd.read_csv("World-happiness-report-updated_2024.csv", encoding='latin-1')
    df2 = pd.read_csv("World-happiness-report-2024.csv", encoding='latin-1')
    
    # Extraindo as regiões do dataset de 2024
    df_regioes = df2[['Country name', 'Regional indicator']]
    
    # Fazendo o merge (Left Join) para trazer o 'Regional indicator' para o histórico
    df = pd.merge(df1, df_regioes, on="Country name", how="left")
    
    return df, df2

try:
    df_history, df_2024 = load_data()
except FileNotFoundError:
    st.error("⚠️ Arquivos CSV não encontrados! Certifique-se de que 'World-happiness-report-updated_2024.csv' e 'World-happiness-report-2024.csv' estão no mesmo diretório.")
    st.stop()

# --- 2. BARRA LATERAL (SIDEBAR) PARA FILTROS ---
st.sidebar.header("Filtros de Análise")
anos_disponiveis = df_history['year'].dropna().unique()
ano_selecionado = st.sidebar.slider("Selecione o Ano Histórico:", int(anos_disponiveis.min()), int(anos_disponiveis.max()), 2023)

regioes_disponiveis = df_history['Regional indicator'].dropna().unique().tolist()
regiao_selecionada = st.sidebar.multiselect("Filtre por Região:", options=regioes_disponiveis, default=regioes_disponiveis[:3])

# Aplicação dos filtros de ano e região
df_filtrado = df_history[(df_history['year'] == ano_selecionado) & (df_history['Regional indicator'].isin(regiao_selecionada))]

# NOVO: Filtro para destacar um país específico
paises_disponiveis = sorted(df_filtrado['Country name'].dropna().unique())
pais_destaque = st.sidebar.selectbox("Destacar País no Gráfico (Opcional):", options=["Nenhum"] + paises_disponiveis)

# --- 3. VISUALIZAÇÃO DE DADOS (EDA) ---
st.header("📊 Exploração de Dados (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Relação: PIB per Capita vs. Felicidade ({ano_selecionado})")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plota todos os países das regiões selecionadas
    sns.scatterplot(
        data=df_filtrado, 
        x='Log GDP per capita', 
        y='Life Ladder', 
        hue='Regional indicator', 
        s=100, alpha=0.7, ax=ax
    )
    
    # NOVO: Se um país for selecionado no filtro, adiciona uma marcação em destaque
    if pais_destaque != "Nenhum":
        dados_pais = df_filtrado[df_filtrado['Country name'] == pais_destaque]
        if not dados_pais.empty:
            # Plota o ponto do país em destaque com estrela grande e borda vermelha
            ax.scatter(
                dados_pais['Log GDP per capita'],
                dados_pais['Life Ladder'],
                color='red',
                s=250,
                marker='*',
                edgecolor='black',
                zorder=5,
                label=f"Destaque: {pais_destaque}"
            )
            # Adiciona o nome do país ao lado do ponto
            for _, row in dados_pais.iterrows():
                ax.annotate(
                    row['Country name'], 
                    (row['Log GDP per capita'], row['Life Ladder']),
                    textcoords="offset points", 
                    xytext=(8, 8), 
                    ha='left', 
                    fontsize=10, 
                    fontweight='bold',
                    color='red'
                )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    # Lista de países por região
    with st.expander("📍 Ver países presentes no gráfico por região"):
        if not df_filtrado.empty:
            paises_por_regiao = df_filtrado.groupby('Regional indicator')['Country name'].unique()
            
            for regiao, paises in paises_por_regiao.items():
                paises_lista = ", ".join(sorted(paises))
                st.markdown(f"**{regiao}:** {paises_lista}")
        else:
            st.info("Nenhum dado encontrado para o filtro selecionado.")

with col2:
    st.subheader("Matriz de Correlação Global")
    cols_numericas = df_history.select_dtypes(include=[np.number]).columns
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_history[cols_numericas].corr(), annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

st.subheader(f"Amostra dos Dados ({ano_selecionado})")
st.dataframe(df_filtrado[['Country name', 'Life Ladder', 'Log GDP per capita', 'Social support']].head(10), use_container_width=True)

# --- 4. MACHINE LEARNING: REGRESSÃO LINEAR BÁSICA ---
st.header("🤖 Modelo Preditivo Básico (Scikit-Learn)")
st.markdown("Treinamento de uma regressão linear para prever o *Life Ladder* com base nos indicadores socioeconômicos (utilizando todos os anos).")

features = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices']
target = 'Life Ladder'

df_ml = df_history[features + [target]].copy()
imputer = SimpleImputer(strategy='median')
df_ml_imputed = pd.DataFrame(imputer.fit_transform(df_ml), columns=df_ml.columns)

X = df_ml_imputed[features]
y = df_ml_imputed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

col_m1, col_m2 = st.columns(2)
col_m1.metric("R² Score (Qualidade do Modelo)", f"{r2_score(y_test, y_pred):.2f}")
col_m2.metric("RMSE (Erro Quadrático Médio)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

st.subheader("Impacto das Variáveis na Felicidade (Coeficientes da Regressão)")
coeficientes = pd.DataFrame({'Variável': features, 'Impacto (Coeficiente)': modelo.coef_})
coeficientes = coeficientes.sort_values(by='Impacto (Coeficiente)', ascending=False)

fig_coef, ax_coef = plt.subplots(figsize=(8, 4))
sns.barplot(data=coeficientes, x='Impacto (Coeficiente)', y='Variável', palette='viridis', ax=ax_coef)
st.pyplot(fig_coef)

# --- 5. INFERÊNCIA INTERATIVA ---
st.header("🔮 Simulador de Felicidade (Inferência em Tempo Real)")
st.markdown("Ajuste os controles deslizantes abaixo com parâmetros fictícios ou reais. O modelo usará esses valores para prever o índice de felicidade (*Life Ladder*).")

col_inf1, col_inf2 = st.columns(2)

with col_inf1:
    input_gdp = st.slider("Log PIB per capita", 
                          min_value=float(X['Log GDP per capita'].min()), 
                          max_value=float(X['Log GDP per capita'].max()), 
                          value=float(X['Log GDP per capita'].mean()),
                          step=0.1)
    
    input_social = st.slider("Suporte Social (0 a 1)", 
                             min_value=float(X['Social support'].min()), 
                             max_value=float(X['Social support'].max()), 
                             value=float(X['Social support'].mean()),
                             step=0.05)

with col_inf2:
    input_health = st.slider("Expectativa de Vida Saudável (Anos)", 
                             min_value=float(X['Healthy life expectancy at birth'].min()), 
                             max_value=float(X['Healthy life expectancy at birth'].max()), 
                             value=float(X['Healthy life expectancy at birth'].mean()),
                             step=1.0)
    
    input_freedom = st.slider("Liberdade para Fazer Escolhas (0 a 1)", 
                              min_value=float(X['Freedom to make life choices'].min()), 
                              max_value=float(X['Freedom to make life choices'].max()), 
                              value=float(X['Freedom to make life choices'].mean()),
                              step=0.05)

if st.button("Calcular Previsão", type="primary"):
    dados_entrada = pd.DataFrame({
        'Log GDP per capita': [input_gdp],
        'Social support': [input_social],
        'Healthy life expectancy at birth': [input_health],
        'Freedom to make life choices': [input_freedom]
    })
    
    previsao = modelo.predict(dados_entrada)[0]
    previsao_formatada = max(0.0, min(10.0, previsao))
    
    st.success(f"🌟 A pontuação de Felicidade prevista (Life Ladder) para esses dados é: **{previsao_formatada:.2f}** / 10")
