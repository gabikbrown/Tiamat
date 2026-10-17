"""Otimização dos hiperparâmetros dos seis pipelines com o optuna.

Uso, sempre a partir da pasta raiz do repositório:

    python -m codigo.otimizar                          # todos os modelos
    python -m codigo.otimizar --modelos knn arvore_decisao    # só alguns
    python -m codigo.otimizar --reiniciar              # apaga os estudos e recomeça
    python -m codigo.otimizar --n-jobs 4               # 4 dobras da validação em paralelo

Para cada algoritmo, o optuna testa configurações até completar 100
conjuntos DISTINTOS de hiperparâmetros (configurações repetidas são
descartadas e não entram na contagem). Cada configuração é avaliada pela
acurácia balanceada média em validação cruzada estratificada de 5 dobras,
calculada APENAS com uma subamostra estratificada de 6.000 registros do
conjunto de treino (ver AMOSTRA_BUSCA em dados.py). O conjunto de teste não
é lido em nenhum momento deste script.

Saídas:
    resultados/otimizacao_optuna.db       banco SQLite com todos os estudos do optuna
    resultados/tentativas_<modelo>.csv    uma linha por configuração testada
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate

from .dados import AMOSTRA_BUSCA, PASTA_RAIZ, SEMENTE, amostra_busca, carregar_treino_teste
from .pipelines import MODELOS, criar_pipeline

N_CONFIGURACOES = 100
N_DOBRAS = 5
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"
CAMINHO_BANCO = PASTA_RESULTADOS / "otimizacao_optuna.db"

# A métrica otimizada é a primeira; as outras são guardadas para análise.
METRICAS = {
    "acuracia_balanceada": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef),
}


def validacao_cruzada_otimizacao():
    """As MESMAS 5 dobras estratificadas são usadas em todas as tentativas e modelos."""
    return StratifiedKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEMENTE)


def url_banco(caminho=CAMINHO_BANCO):
    """Endereço do banco SQLite no formato aceito pelo optuna (funciona no Linux e no Windows)."""
    return f"sqlite:///{Path(caminho).resolve().as_posix()}"


def _tentativa_repetida(trial):
    """Devolve o número de uma tentativa concluída com os mesmos hiperparâmetros (ou None)."""
    for anterior in trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if anterior.params == trial.params:
            return anterior.number
    return None


def criar_objetivo(nome_modelo, X_treino, y_treino, n_jobs=1):
    """Cria a função objetivo do optuna para um dos algoritmos."""
    cv = validacao_cruzada_otimizacao()

    def objetivo(trial):
        pipeline = criar_pipeline(trial, nome_modelo)  # aqui o optuna sorteia os hiperparâmetros

        repetida = _tentativa_repetida(trial)
        if repetida is not None:
            raise optuna.TrialPruned(f"configuração idêntica à da tentativa {repetida}")

        resultado = cross_validate(
            pipeline, X_treino, y_treino, cv=cv, scoring=METRICAS, n_jobs=n_jobs, error_score="raise"
        )
        acuracias = resultado["test_acuracia_balanceada"]
        trial.set_user_attr("acuracia_balanceada_dobras", [round(float(v), 6) for v in acuracias])
        trial.set_user_attr("acuracia_balanceada_dp", float(np.std(acuracias, ddof=1)))
        trial.set_user_attr("roc_auc_media", float(np.mean(resultado["test_roc_auc"])))
        trial.set_user_attr("mcc_media", float(np.mean(resultado["test_mcc"])))
        trial.set_user_attr("tempo_ajuste_medio_s", float(np.mean(resultado["fit_time"])))
        return float(np.mean(acuracias))

    return objetivo


def otimizar_modelo(nome_modelo, X_treino, y_treino, armazenamento, n_configuracoes=N_CONFIGURACOES,
                    n_jobs=1, reiniciar=False):
    """Roda (ou continua) o estudo do optuna de um algoritmo até ter n_configuracoes concluídas."""
    if reiniciar:
        try:
            optuna.delete_study(study_name=nome_modelo, storage=armazenamento)
        except KeyError:
            pass  # o estudo ainda não existia

    # TPE multivariado com decomposição em grupos: modela em conjunto os
    # hiperparâmetros que aparecem juntos, o que lida bem com o espaço
    # condicional (k só existe quando a redução é SelectKBest, por exemplo).
    amostrador = optuna.samplers.TPESampler(seed=SEMENTE, multivariate=True, group=True)
    estudo = optuna.create_study(
        study_name=nome_modelo, storage=armazenamento, direction="maximize",
        sampler=amostrador, load_if_exists=True,
    )
    estudo.set_user_attr("modelo", MODELOS[nome_modelo])
    estudo.set_user_attr("metrica", "acurácia balanceada média em validação cruzada estratificada de 5 dobras "
                                    f"sobre subamostra de {AMOSTRA_BUSCA} registros do treino")

    concluidas = len(estudo.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if concluidas < n_configuracoes:
        estudo.optimize(
            criar_objetivo(nome_modelo, X_treino, y_treino, n_jobs=n_jobs),
            n_trials=3 * n_configuracoes,  # teto de segurança; quem encerra é o callback abaixo
            callbacks=[optuna.study.MaxTrialsCallback(n_configuracoes, states=(TrialState.COMPLETE,))],
        )
    return estudo


def exportar_tentativas(estudo, pasta=PASTA_RESULTADOS):
    """Salva um CSV com todas as tentativas do estudo (para consulta sem abrir o banco)."""
    tabela = estudo.trials_dataframe(attrs=("number", "value", "duration", "params", "user_attrs", "state"))
    tabela["duration"] = tabela["duration"].dt.total_seconds()
    tabela = tabela.rename(columns={"number": "tentativa", "value": "acuracia_balanceada_media",
                                    "duration": "duracao_s", "state": "estado"})
    caminho = pasta / f"tentativas_{estudo.study_name}.csv"
    tabela.to_csv(caminho, index=False)
    return caminho


def main():
    parser = argparse.ArgumentParser(description="Otimização de hiperparâmetros (projeto Tiamat).")
    parser.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS),
                        help="algoritmos a otimizar (padrão: todos)")
    parser.add_argument("--n-configuracoes", type=int, default=N_CONFIGURACOES,
                        help="número de configurações distintas por algoritmo (padrão: 100)")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="dobras da validação cruzada rodando em paralelo (padrão: CPUs do Slurm ou 1)")
    parser.add_argument("--banco", default=str(CAMINHO_BANCO), help="arquivo SQLite do optuna")
    parser.add_argument("--reiniciar", action="store_true", help="apaga os estudos existentes antes de começar")
    args = parser.parse_args()

    # Avisos esperados e inofensivos: alguns pontos do espaço de busca (por
    # exemplo, regularização muito fraca) atingem o limite de iterações, e o
    # argumento group do TPE é marcado como experimental pelo optuna.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    X_treino, X_teste, y_treino, y_teste = carregar_treino_teste()
    del X_teste, y_teste  # o teste não participa da otimização
    X_busca, y_busca = amostra_busca(X_treino, y_treino)
    print(f"Treino completo: {X_treino.shape[0]} registros | subamostra da busca: {X_busca.shape[0]} "
          f"({y_busca.mean():.1%} acima de 50 mil). n_jobs = {args.n_jobs}", flush=True)

    armazenamento = url_banco(args.banco)
    for nome in args.modelos:
        print(f"\n===== {MODELOS[nome]} =====", flush=True)
        inicio = time.perf_counter()
        estudo = otimizar_modelo(nome, X_busca, y_busca, armazenamento, args.n_configuracoes,
                                 n_jobs=args.n_jobs, reiniciar=args.reiniciar)
        duracao = time.perf_counter() - inicio
        concluidas = len(estudo.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        descartadas = len(estudo.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))
        print(f"Concluídas: {concluidas} | repetidas descartadas: {descartadas} | tempo: {duracao / 60:.1f} min")
        print(f"Melhor acurácia balanceada (CV): {estudo.best_value:.4f}")
        print(f"Melhores hiperparâmetros: {estudo.best_params}")
        print(f"Tabela salva em {exportar_tentativas(estudo)}", flush=True)


if __name__ == "__main__":
    main()
