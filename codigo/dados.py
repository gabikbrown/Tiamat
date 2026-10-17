"""Leitura, limpeza e divisão do conjunto de dados Adult (Census Income).

O arquivo original (adult.data) vem do UCI Machine Learning Repository e não
tem cabeçalho: são 15 colunas separadas por vírgula, sendo as 14 primeiras
atributos de uma pessoa e a última a renda anual (">50K" ou "<=50K"). Valores
ausentes aparecem como "?".

Este arquivo é apenas o conjunto de TREINO original do UCI (32.561 linhas).
O conjunto de teste do projeto sai daí, por uma divisão estratificada.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Semente usada em TODAS as etapas com aleatoriedade (divisão dos dados,
# subamostra da busca, validação cruzada, modelos e amostrador do optuna).
SEMENTE = 42

# Fração dos dados reservada para o teste final (nunca usada na otimização).
FRACAO_TESTE = 0.20

# Tamanho da subamostra estratificada do treino usada na busca de hiperparâmetros.
# Ajustar um SVM com kernel RBF em ~24 mil linhas leva cerca de 30 s por dobra;
# com 100 configurações por algoritmo, a busca completa levaria horas em 2 CPUs.
AMOSTRA_BUSCA = 6000

PASTA_RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = PASTA_RAIZ / "dados" / "adult.data"

# Nomes das 15 colunas do arquivo original, na ordem do UCI.
COLUNAS = [
    "age", "workclass", "fnlwgt", "education", "educational-num", "marital-status",
    "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]

# fnlwgt é o peso amostral do censo (quantas pessoas da população cada registro
# representa), não uma característica da pessoa, por isso não é usado como atributo.
ATRIBUTOS_NUMERICOS = [
    "age", "educational-num", "capital-gain", "capital-loss", "hours-per-week",
]
ATRIBUTOS_CATEGORICOS = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "gender", "native-country",
]
ATRIBUTOS = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS  # 13 atributos

# Descrição de cada atributo (tradução livre da documentação do UCI).
DESCRICOES = {
    "age": "Idade em anos",
    "educational-num": "Anos de estudo (escala numérica do censo)",
    "capital-gain": "Ganho de capital no ano (dólares)",
    "capital-loss": "Perda de capital no ano (dólares)",
    "hours-per-week": "Horas trabalhadas por semana",
    "workclass": "Tipo de vínculo de trabalho (privado, público, autônomo etc.)",
    "education": "Nível de escolaridade (categórico)",
    "marital-status": "Estado civil",
    "occupation": "Ocupação",
    "relationship": "Papel na família (esposo, filho, não aparentado etc.)",
    "race": "Raça autodeclarada",
    "gender": "Sexo (male/female, conforme o censo)",
    "native-country": "País de origem",
}

NOME_ALVO = "income"
NOME_ALVO_BINARIO = "rendimento_acima_50k"
ROTULOS_CLASSES = {0: "<=50K (renda até 50 mil dólares)", 1: ">50K (renda acima de 50 mil dólares)"}


def ler_dados_brutos(caminho=CAMINHO_DADOS):
    """Lê o arquivo original e devolve um DataFrame com os nomes das colunas.

    ``skipinitialspace`` remove o espaço que o UCI coloca depois de cada vírgula,
    e ``na_values="?"`` transforma os valores ausentes em NaN.
    """
    return pd.read_csv(
        caminho, header=None, names=COLUNAS, skipinitialspace=True, na_values="?"
    )


def preparar_dados(df_bruto):
    """Remove registros com valores ausentes e duplicatas, e separa X e y.

    Linhas com ``?`` são removidas (cerca de 7 % dos registros). Linhas
    IDÊNTICAS em todas as 15 colunas são tratadas como duplicatas de digitação
    e mantidas apenas uma vez, antes da divisão treino/teste. Registros com os
    mesmos 13 atributos mas valores diferentes em fnlwgt são pessoas distintas
    no censo e são mantidos.

    O alvo é codificado como 1 para renda acima de 50 mil dólares e 0 caso contrário.
    """
    df = df_bruto.dropna().drop_duplicates().reset_index(drop=True)
    X = df[ATRIBUTOS].copy()
    y = (df[NOME_ALVO].str.strip() == ">50K").astype(int).rename(NOME_ALVO_BINARIO)
    return X, y


def dividir_treino_teste(X, y):
    """Divisão estratificada 80/20, mantendo a proporção de rendas nos dois conjuntos."""
    return train_test_split(X, y, test_size=FRACAO_TESTE, stratify=y, random_state=SEMENTE)


def carregar_treino_teste():
    """Atalho usado pelo notebook e pelo script de otimização: lê, limpa e divide os dados."""
    X, y = preparar_dados(ler_dados_brutos())
    return dividir_treino_teste(X, y)


def amostra_busca(X_treino, y_treino, n=AMOSTRA_BUSCA):
    """Subamostra estratificada do treino, usada somente na busca de hiperparâmetros."""
    if n >= len(X_treino):
        return X_treino, y_treino
    _, indices = train_test_split(
        range(len(X_treino)), test_size=n, stratify=y_treino, random_state=SEMENTE
    )
    indices = sorted(indices)
    return X_treino.iloc[indices], y_treino.iloc[indices]
