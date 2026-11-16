

![Imagem](Cabeçalho.png)

<div align="center">

# **Previsão de renda acima de 50 mil dólares com aprendizado de máquina**
</div>

**Autora:** Gabriela O. de Lima Cabral\
**Professor:** Daniel Roberto Cassar\
**Ilum - Escola de Ciência**\
**Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)**


## Sobre o trabalho

Decisões sobre crédito, contratação e políticas públicas costumam depender de estimativas de renda. Este trabalho investiga se é possível prever se uma pessoa ganha mais de 50 mil dólares por ano a partir de características como idade, escolaridade, ocupação, estado civil e horas trabalhadas por semana. Os dados vêm do conjunto **Adult**, do UCI Machine Learning Repository, extraído do censo dos Estados Unidos de 1994. São 32.561 registros, e dos 14 atributos originais, 12 foram usados. Cerca de 24% das pessoas têm renda acima de 50 mil dólares, e por isso a métrica principal é a área sob a curva precisão × sensibilidade (**PR-AUC**), e não a acurácia, que seria enganosa com classes desbalanceadas.

Foram comparados cinco algoritmos de famílias diferentes (regressão logística, *k* vizinhos mais próximos, árvore de decisão, floresta aleatória e *gradient boosting*) com um modelo de referência (*baseline*) que sempre prevê a classe majoritária. Cada algoritmo foi montado como um *pipeline* do scikit-learn, com o pré-processamento e os hiperparâmetros otimizados pelo Optuna em 100 tentativas distintas. Um cuidado central do trabalho foi evitar vazamento de dados: o conjunto tem muitas pessoas com perfis idênticos ou quase idênticos, então exemplos parecidos foram agrupados, e cada grupo ficou inteiro no treino ou inteiro no teste. O conjunto de teste foi usado uma única vez, depois de o modelo campeão já ter sido escolhido.

O *gradient boosting* foi o melhor *pipeline*, com PR-AUC de 0,836 no teste (intervalo de confiança de 95% de 0,822 a 0,851), contra 0,238 do *baseline*. A floresta aleatória ficou em segundo lugar, com 0,819. Os atributos que mais pesaram nas previsões foram ganho de capital, idade, escolaridade, estado civil, perda de capital e ocupação.

## Principais resultados

A tabela mostra a PR-AUC de cada *pipeline*. A coluna da validação cruzada é a média dos 5 *folds* de uma validação cruzada feita só com o conjunto de treino. A coluna do teste vem de uma única avaliação, com intervalo de confiança de 95% calculado por *bootstrap*.

| Pipeline | Validação cruzada (treino) | Teste | IC 95% (teste) |
|---|:---:|:---:|:---:|
| *Baseline* | 0,242 | 0,238 | 0,227 a 0,249 |
| Regressão logística | 0,739 | 0,745 | 0,722 a 0,769 |
| *k*-NN | 0,751 | 0,760 | 0,738 a 0,781 |
| Árvore de decisão | 0,777 | 0,783 | 0,766 a 0,801 |
| Floresta aleatória | 0,809 | 0,819 | 0,803 a 0,835 |
| ***Gradient boosting*** | **0,826** | **0,836** | **0,822 a 0,851** |

A PR-AUC vai de 0 a 1. Um modelo que chuta ao acaso fica perto da proporção de positivos (cerca de 0,24), e por isso o *baseline* serve de piso para a comparação.

## Estrutura do repositório

| Arquivo ou pasta | Conteúdo |
|---|---|
| `ML_Tiamat.ipynb` | Notebook com todo o trabalho: código, gráficos, discussão, conclusões, descrição do uso de IA e referências. É o arquivo principal. |
| `adult.data` | Conjunto de dados Adult, no arquivo de treino original do UCI (32.561 registros, sem linha de cabeçalho). |
| `estudos/` | Estudos do Optuna salvos em SQLite, um arquivo `.db` por *pipeline* (`regressao_logistica`, `knn`, `arvore_decisao`, `floresta_aleatoria` e `gradient_boosting`). Guardam todas as tentativas da otimização. |
| `requirements.txt` | Bibliotecas e versões usadas. |
| `Cabeçalho.png` | Imagem do cabeçalho deste README. |
| `LICENSE` | Licença GPL-3.0. |
| `.gitignore` | Arquivos que o git não deve versionar, como caches e *checkpoints* do Jupyter. |

## Como navegar pelo notebook

O notebook segue a ordem em que o trabalho foi feito. Para uma leitura rápida, vale começar pela Introdução, ir para a Seção 10, com o resultado no teste, e terminar nas Conclusões.

| Seção | O que contém |
|---|---|
| 1. Introdução | O problema, o objetivo e a escolha da métrica principal. |
| 2. Bibliotecas e configurações | Importações, semente aleatória (24) e parâmetros gerais. |
| 3. O conjunto de dados | Dicionário dos atributos, valores ausentes, duplicatas e a remoção de `fnlwgt` e `education`. |
| 4. Divisão treino/teste | Divisão agrupada e estratificada e a conferência de que ela não vaza dados. |
| 5. Análise exploratória | Feita só com o treino: desbalanceamento das classes, atributos numéricos e categóricos e valores ausentes. |
| 6. Estratégia de avaliação | Métricas e validação cruzada estratificada e agrupada. |
| 7. Pré-processamento e *baseline* | O *pipeline* de pré-processamento e o modelo de referência. |
| 8. Otimização com Optuna | Os cinco *pipelines*, com os intervalos de busca de cada hiperparâmetro e as justificativas. |
| 9. Comparação por validação cruzada | Nova validação cruzada dos cinco *pipelines* otimizados e escolha do campeão. |
| 10. Avaliação no teste | Uso único do conjunto de teste, com intervalos de confiança por *bootstrap*. |
| 11. Explicabilidade | Importância por permutação: quais atributos o modelo campeão mais usa. |
| Conclusões | Resultado principal e aprendizados. |
| Uso de IA e Referências | Como a IA foi usada no trabalho e as fontes consultadas. |

## Como reproduzir

O trabalho foi feito com Python 3.13. Os comandos abaixo são digitados no terminal.

1. **Baixar o repositório.** `git clone https://github.com/gabikbrown/Tiamat.git` cria uma cópia local de todos os arquivos, incluindo os dados e os estudos do Optuna. Em seguida, `cd Tiamat` entra na pasta.
2. **Criar um ambiente virtual (opcional, mas recomendado).** `python -m venv .venv` cria um ambiente isolado, para que as bibliotecas instaladas aqui não interfiram em outros projetos. Para ativá-lo, use `source .venv/bin/activate` no Linux ou macOS, ou `.venv\Scripts\activate` no Windows.
3. **Instalar as bibliotecas.** `pip install -r requirements.txt` instala as versões listadas no arquivo.
4. **Executar o notebook.** Abra `ML_Tiamat.ipynb` no VS Code, com a extensão Jupyter, ou no Jupyter (que pode ser instalado com `pip install notebook`), e execute todas as células em ordem.

Sobre o tempo de execução: a otimização carrega os estudos que já estão na pasta `estudos/` e só roda as tentativas que faltam para chegar a 100 por *pipeline*. Com a pasta presente, o notebook reaproveita as tentativas salvas e termina bem mais rápido. Sem ela, a otimização recomeça do zero e leva perto de uma hora, dependendo do computador. Todas as etapas com aleatoriedade usam a semente 24, para que os resultados possam ser reproduzidos.


## Autora
<table>
  <tr>
     <td align="center">
      <a href="https://github.com/gabikbrown">
        <img src="https://github.com/gabikbrown.png" width="100" alt="Foto de Gabriela Cabral"><br>
        <sub><b>Gabriela Cabral</b></sub>
      </a>
    </td>
 </tr>
</table>

## Professor 
<table>
  <tr>
    <td align="center">
      <a href="#" title="Prof. Daniel R. Cassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do Daniel do Github"/><br>
          <a href="https://github.com/drcassar"><b>Prof. Dr. Daniel R. Cassar<b></a>
      </a>
    </td>
  </tr>
</table>

## Fonte dos dados

BECKER, B.; KOHAVI, R. *Adult*. UCI Machine Learning Repository, 1996. DOI: [10.24432/C5XW20](https://doi.org/10.24432/C5XW20).

## Licença

Este repositório é distribuído sob a licença GPL-3.0. Veja o arquivo [`LICENSE`](LICENSE).
