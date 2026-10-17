"""Pipelines e espaços de busca de hiperparâmetros usados pelo optuna.

Cada pipeline tem três etapas e TODAS as escolhas de normalização, redução e
classificação são feitas pelo optuna:

1. ``preprocessamento``: os atributos numéricos são normalizados (padronização
   z-score ou transformação por quantis, escolhida pelo optuna) e os atributos
   categóricos viram colunas binárias (one-hot);
2. ``reducao``: nenhuma, seleção dos k melhores atributos (SelectKBest) ou
   análise de componentes principais (PCA);
3. ``classificador``: um dos seis algoritmos comparados no trabalho.

Como o pipeline inteiro é ajustado dentro de cada dobra da validação cruzada,
a normalização, a codificação e a redução "aprendem" apenas com os dados de
treino daquela dobra, o que evita vazamento de informação.

As justificativas dos intervalos estão nos comentários de cada função e,
com mais detalhes, na seção "Espaço de busca dos hiperparâmetros" do notebook.
"""

from functools import partial

import optuna
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .dados import ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS, SEMENTE

# Depois da codificação one-hot dos 8 atributos categóricos, a matriz tem cerca de
# 101 colunas (5 numéricas e 96 binárias; o número varia um pouco entre dobras,
# por causa das categorias raras). O limite superior da redução fica em 80
# colunas, abaixo desse total, para que k e o número de componentes nunca
# excedam o que está disponível.
MAX_COLUNAS_REDUCAO = 80

# Identificador usado no banco do optuna -> nome por extenso
MODELOS = {
    "regressao_logistica": "Regressão logística",
    "knn": "k vizinhos mais próximos (k-NN)",
    "arvore_decisao": "Árvore de decisão",
    "floresta_aleatoria": "Floresta aleatória",
    "gradient_boosting": "Gradient boosting com histogramas",
    "lightgbm": "LightGBM (gradient boosting com histogramas)",
}

# Nomes curtos para legendas de gráficos
NOMES_CURTOS = {
    "regressao_logistica": "Reg. logística",
    "knn": "k-NN",
    "arvore_decisao": "Árvore de decisão",
    "floresta_aleatoria": "Floresta aleatória",
    "gradient_boosting": "Gradient boosting",
    "lightgbm": "LightGBM",
}


# ---------------------------------------------------------------------------
# Etapas comuns a todos os pipelines
# ---------------------------------------------------------------------------
def sugerir_normalizador(trial):
    """Padronização (média 0, desvio 1) ou transformação por quantis, para os atributos numéricos.

    Ganho de capital e perda de capital são quase sempre zero, com poucos valores
    muito altos. A transformação por quantis leva cada atributo a uma distribuição
    aproximadamente normal e reduz a influência desses extremos nas distâncias
    usadas pelo k-NN.
    """
    tipo = trial.suggest_categorical("normalizador", ["padronizacao", "quantis"])
    if tipo == "padronizacao":
        return StandardScaler()
    return QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEMENTE)


def criar_preprocessamento(trial):
    """Normaliza os atributos numéricos e codifica os categóricos em colunas binárias.

    ``handle_unknown="ignore"`` faz com que uma categoria ausente no treino da
    dobra (ou no teste) vire um vetor de zeros, em vez de provocar erro.
    """
    return ColumnTransformer(
        [
            ("numericos", sugerir_normalizador(trial), ATRIBUTOS_NUMERICOS),
            (
                "categoricos",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ATRIBUTOS_CATEGORICOS,
            ),
        ]
    )


def sugerir_reducao(trial):
    """Nenhuma redução, seleção de atributos (SelectKBest) ou PCA.

    k e o número de componentes variam de 5 a 80 (ver MAX_COLUNAS_REDUCAO).
    """
    tipo = trial.suggest_categorical("reducao", ["nenhuma", "selecao_kbest", "pca"])
    if tipo == "nenhuma":
        return "passthrough"
    if tipo == "selecao_kbest":
        k = trial.suggest_int("kbest_k", 5, MAX_COLUNAS_REDUCAO)
        criterio = trial.suggest_categorical("kbest_criterio", ["anova_f", "informacao_mutua"])
        if criterio == "anova_f":
            funcao = f_classif
        else:
            funcao = partial(mutual_info_classif, random_state=SEMENTE)
        return SelectKBest(score_func=funcao, k=k)
    n_componentes = trial.suggest_int("pca_n_componentes", 2, MAX_COLUNAS_REDUCAO)
    return PCA(n_components=n_componentes, random_state=SEMENTE)


# ---------------------------------------------------------------------------
# Classificadores e seus hiperparâmetros
# ---------------------------------------------------------------------------
def _regressao_logistica(trial):
    # C é o inverso da força de regularização; escala log porque seu efeito é
    # multiplicativo. l1_ratio = 0 equivale à penalidade L2 (ridge), 1 à L1
    # (lasso) e valores intermediários à elastic net. O solver saga aceita os três casos.
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-4, 1e3, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        solver="saga",
        max_iter=5000,
        random_state=SEMENTE,
    )


def _knn(trial):
    # k entre 1 (fronteira muito irregular) e 100. Em cada dobra há cerca de
    # 4.800 exemplos de treino; a regra prática k ~ raiz de n (~69) fica no
    # intervalo, perto do meio na escala logarítmica.
    return KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 1, 100),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_categorical("p", [1, 2]),  # 1 = Manhattan, 2 = euclidiana
    )


def _arvore_decisao(trial):
    # Uma única árvore: max_depth e min_samples_leaf controlam a complexidade e
    # ccp_alpha poda ramos que pouco melhoram o ajuste. Árvores isoladas tendem
    # a se ajustar demais, por isso a poda é escolhida pelo optuna.
    return DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        ccp_alpha=trial.suggest_float("ccp_alpha", 1e-5, 1e-1, log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=SEMENTE,
    )


def _floresta_aleatoria(trial):
    # max_features (fração de atributos sorteada em cada divisão) costuma ser o
    # hiperparâmetro mais influente da floresta; profundidade e tamanho mínimo
    # das folhas controlam o quanto cada árvore pode se ajustar aos dados.
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_float("max_features", 0.05, 1.0),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        random_state=SEMENTE,
        n_jobs=1,
    )


def _gradient_boosting(trial):
    # Taxa de aprendizado e número de iterações se compensam (taxa menor exige
    # mais árvores); max_leaf_nodes controla a complexidade de cada árvore e
    # l2_regularization penaliza valores grandes nas folhas.
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.5, log=True),
        max_iter=trial.suggest_int("max_iter", 50, 600, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 60),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        early_stopping=False,
        random_state=SEMENTE,
    )


def _lightgbm(trial):
    # LightGBM constrói as árvores por folhas (num_leaves), com amostragem de
    # linhas (subsample) e de atributos (colsample_bytree) em cada árvore. Taxa de
    # aprendizado e número de árvores se compensam, e reg_lambda penaliza as folhas.
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.5, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=SEMENTE,
        n_jobs=1,
        verbose=-1,
    )


SUGERIR_CLASSIFICADOR = {
    "regressao_logistica": _regressao_logistica,
    "knn": _knn,
    "arvore_decisao": _arvore_decisao,
    "floresta_aleatoria": _floresta_aleatoria,
    "gradient_boosting": _gradient_boosting,
    "lightgbm": _lightgbm,
}


def criar_pipeline(trial, nome_modelo):
    """Monta o pipeline preprocessamento -> redução -> classificador para uma tentativa do optuna."""
    return Pipeline([
        ("preprocessamento", criar_preprocessamento(trial)),
        ("reducao", sugerir_reducao(trial)),
        ("classificador", SUGERIR_CLASSIFICADOR[nome_modelo](trial)),
    ])


def pipeline_com_parametros(nome_modelo, parametros):
    """Recria o pipeline a partir de um dicionário de hiperparâmetros já escolhidos.

    ``optuna.trial.FixedTrial`` devolve os valores do dicionário em vez de
    sorteá-los, então o mesmo código que define o espaço de busca serve para
    reconstruir a melhor configuração encontrada.
    """
    return criar_pipeline(optuna.trial.FixedTrial(parametros), nome_modelo)
