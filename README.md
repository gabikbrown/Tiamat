![Imagem](Cabeçalho.png)

<div align="center">
  
# **Previsão de renda acima de 50 mil dólares com aprendizado de máquina**
</div>

**Autora:** Gabriela O. de Lima Cabral\
**Professor:** Daniel Roberto Cassar\
**Ilum - Escola de Ciência**\
**Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)**

## Introdução e Objetivos
Decisões sobre crédito, contratação e políticas públicas costumam depender de estimativas de renda. Este trabalho usa o conjunto **Adult** (censo dos Estados Unidos, 1994) para estudar se é possível prever se uma pessoa ganha mais de 50 mil dólares por ano, a partir de 13 atributos como idade, escolaridade, ocupação, estado civil e horas trabalhadas por semana.

O objetivo é comparar seis algoritmos de famílias diferentes (regressão logística, k vizinhos mais próximos, árvore de decisão, floresta aleatória, gradient boosting e SVM) com um modelo de referência, otimizando os hiperparâmetros de cada um com o optuna. A pergunta central é se o melhor modelo generaliza para dados que não foram usados em nenhuma decisão. Como a classe de alta renda é minoritária, a métrica principal é a **acurácia balanceada**.

## Dados
O arquivo `dados/adult.data` tem 32.561 registros, com 14 atributos e a renda anual como alvo. 

## Metodologia

## Experimentos e Resultado

## Estrutura do presente repositório
Todo o trabalho está no notebook ML_Tiamat.ipynb, que contém a introdução, a análise, os experimentos, os resultados e as conclusões. Ele lê o arquivo dados/adult.data e não depende de nenhum outro arquivo do projeto. 

```
Tiamat/
├── ML_Tiamat.ipynb      relatório completo 
├── README.md         este arquivo
└── adult.data    conjunto de treino do Adult (32.561 registros)
```

## Como executar
1. Crie um ambiente com Python 3.13 e instale as bibliotecas usadas, com as versões fixadas em `requirements.txt`:
```bash
   pip install -r requirements.txt
```
2. Abra o `Tiamat.ipynb` a partir da pasta raiz do repositório e execute todas as células, de cima para baixo. 

## Uso de Inteligência Artificial 
Este trabalho foi desenvolvido com o auxílio do assistente de IA Claude (Anthropic). 
## Referências
