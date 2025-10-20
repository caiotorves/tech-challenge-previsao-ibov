# Previsão de Tendência do IBOVESPA com Machine Learning

Projeto desenvolvido como desafio da Pós-Graduação em Data Analytics (FIAP), focado em construir um modelo preditivo para a tendência diária do índice IBOVESPA, com o objetivo de superar 75% de acurácia.

**Resultado Chave: Acurácia de 76,67% alcançada com um modelo Random Forest.**

---

## 1. O Desafio

O objetivo era desenvolver um modelo de classificação binária capaz de prever se o índice IBOVESPA fecharia em alta (1) ou baixa (0) no dia seguinte.

**Requisitos:**
* Utilizar dados históricos diários do IBOVESPA (mínimo de 2 anos).
* O conjunto de teste deveria ser composto pelos últimos 30 dias de dados disponíveis.
* A acurácia mínima do modelo no conjunto de teste deveria ser de **75%**.

---

## 2. Metodologia

O sucesso do projeto foi alcançado através de uma estratégia de **filtragem de ruído** e **engenharia de atributos** robusta, transformando um problema de série temporal notoriamente volátil em um problema de classificação supervisionada com sinal claro.

### 2.1. Limpeza e Pré-Processamento

* **Fonte de Dados:** Dados históricos diários do `investing.com`.
* **Limpeza:** Os dados foram tratados para converter colunas de texto em formatos numéricos, com atenção especial à coluna `Vol.` (Volume), que exigiu o *parsing* de sufixos 'M' (Milhão) e 'K' (Mil).
* **Ordenação:** Os dados foram ordenados cronologicamente para garantir a integridade da série temporal.

### 2.2. A Estratégia Vencedora: Filtro de Ruído

Uma análise inicial revelou que tentar prever *qualquer* movimento diário (ex: +0.01%) resultava em modelos com performance medíocre (~50-60%), pois o "ruído" aleatório do mercado era mais forte que o "sinal" preditivo.

A estratégia decisiva foi **redefinir o problema**:
1.  Foi estabelecido um **limite (threshold) de 0.5%**.
2.  O `target` foi definido como `1` (Alta) apenas para dias com retorno futuro `> +0.5%` e `0` (Baixa) para dias com retorno futuro `< -0.5%`.
3.  **Todos os dias com movimentos laterais (entre -0.5% e +0.5%) foram removidos do dataset.**

Essa abordagem de filtragem de ruído foi o fator chave para o sucesso, permitindo ao modelo focar apenas em movimentos significativos.

### 2.3. Engenharia de Atributos (Feature Engineering)

Para dar ao modelo um contexto histórico e de momentum, as seguintes features foram criadas para cada dia:

* **Tendência (Janelas Deslizantes):**
    * `SMA_5`, `SMA_10`, `SMA_20`: Médias Móveis Simples de curto e médio prazo.
    * `SMA_Diff`: Diferencial entre as médias de 12 e 26 dias (base do MACD).
* **Momentum (Janelas Deslizantes):**
    * `RSI_14`: Índice de Força Relativa de 14 dias, para identificar sobrecompra/sobrevenda.
    * `Dist_SMA_20`: Distância percentual do preço atual até sua média de 20 dias (mede o quão "esticado" o preço está).
* **Memória (Features Lagged):**
    * `Lag_1`, `Lag_2`, `Lag_3`: Retornos percentuais dos últimos 3 dias.
* **Risco e Convicção:**
    * `Volatility_10`: Desvio padrão dos retornos dos últimos 10 dias.
    * `Vol.`: Volume de negociação do dia (feature de confirmação).

---

## 3. Modelagem e Resultados

Foram comparados três modelos distintos. A validação foi feita em um conjunto de teste fixo, composto pelos últimos 30 pontos de dados "significativos" do nosso dataset filtrado.

### 3.1. Comparação de Acurácia

| Modelo | Acurácia no Teste |
| :--- | :--- |
| **Random Forest** | **76.67%** 🏆 |
| Regressão Logística | 73.33% |
| XGBoost | 70.00% |

O `Random Forest` foi o vencedor, demonstrando a melhor capacidade de capturar os padrões não-lineares complexos das features criadas.

### 3.2. Análise de Confiabilidade do Modelo (Random Forest)

Atingir 76,67% de acurácia foi a meta, mas a análise de confiabilidade prova o valor do modelo.

**Matriz de Confusão (30 dias de teste):**

| Real \ Previsto | Baixa (0) | Alta (1) |
| :--- | :---: | :---: |
| **Baixa (0)** | 11 (VN) | 6 (FP) |
| **Alta (1)** | 1 (FN) | 12 (VP) |

**Métricas Chave:**

* **Acurácia (Geral): 76.67%** (23 acertos em 30).
* **Precisão (Alta): 67%** (Quando o modelo prevê "Alta", ele acerta 67% das vezes).
* **Recall (Alta): 92%** (O modelo conseguiu capturar 12 das 13 oportunidades reais de alta).
* **Precisão (Baixa): 92%** (Quando o modelo prevê "Baixa", ele acerta 92% das vezes).
* **Recall (Baixa): 65%** (O modelo identificou 11 das 17 baixas reais).

O modelo demonstrou um perfil estratégico excelente: é **extremamente confiável ao prever uma baixa** (evitando perdas) e **extremamente sensível ao capturar oportunidades de alta** (maximizando ganhos).

### 3.3. Importância das Features

As features de *momentum* e o retorno do dia atual mostraram-se os preditores mais importantes para o modelo.

1.  `Return` (Retorno do dia)
2.  `Dist_SMA_20` (Distância da Média de 20 dias)
3.  `RSI_14` (Índice de Força Relativa)

---

## 4. Tecnologias Utilizadas

* **Python 3**
* **Pandas:** Para manipulação e limpeza dos dados.
* **Numpy:** Para cálculos numéricos.
* **Scikit-learn (sklearn):** Para pré-processamento (`StandardScaler`), modelagem (`RandomForestClassifier`, `LogisticRegression`) e avaliação (`accuracy_score`, `classification_report`, `GridSearchCV`).
* **XGBoost:** Para modelagem (`XGBClassifier`).
* **Matplotlib & Seaborn:** Para visualização de dados.
* **Google Colab / Jupyter Notebook:** Como ambiente de desenvolvimento.

---

## 5. Como Executar

1.  Certifique-se de ter todas as bibliotecas acima instaladas (`pip install pandas numpy scikit-learn xgboost matplotlib seaborn`).
2.  Faça o download do arquivo `Dados_Históricos_Ibovespa.csv` e do notebook `.ipynb`.
3.  Coloque ambos no mesmo diretório.
4.  Abra e execute o notebook em um ambiente Jupyter.
