#santander-satisfaction

## Solução para o desafio "Santander Customer Satisfaction" no Kaggle

## 1. Introdução

Da página da [competição](https://www.kaggle.com/c/santander-customer-satisfaction): 

"From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience."

O conjunto de dados fornecido continha 371 variáveis, 76.020 exemplos de treino e 75.818 exemplos de teste. A variável alvo "TARGET" carregava uma flag informando se dado cliente estava insatisfeito ou não. Este foi meu primeiro desafio "Featured" (classe de problemas que têm prêmios em dinheiro e contam pontos para se tornar Kaggle Master)  que participei. Foi a maior competição da história do Kaggle em termos de adesão, com 5.123 times.

Foi o melhor desempenho em competições que obtive até hoje. A minha melhor submissão obteve AUC de 0.826928 na leaderboard privada, me colocando na posição 470 de 5.123 e rendendo um "troféu" de Top 10%. Neste documento vou mostrar algumas particularidades interessantes da competição e as principais técnicas utilizadas na minha solução.

## 2. Análise exploratória, feature engineering

O conjunto de dados fornecido continha  371 variáveis, 76.020 exemplos de treino e 75.818 exemplos de teste. A variável alvo "TARGET" carregava uma flag informando se dado cliente estava insatisfeito ou não. Na pasta `/notebooks` temos dois notebooks em Python 2 (`data-analysis` e `feature-engineering`) dedicados à limpeza e visualização dos dados e feature engineering.

O primeiro ponto a notar nos dados é o desequilíbrio entre as classes na variável alvo. Somente 3.95% dos campos target pertenciam à classe positiva (cliente insatisfeito).

Além disso, a natureza do problema trazia um desafio para a modelagem: usuários do Kaggle reportaram exemplos iguais para ambas as classes. Não havia separação clara. Isso pode ser visualizado no gráfico a seguir, que mostra as duas dimensões mais importantes após a apliação de um PCA nos dados (detalhes [aqui](https://www.kaggle.com/inversion/santander-customer-satisfaction/pca-visualization/run/175633/output)):

![](https://github.com/gdmarmerola/santander-satisfaction/blob/master/pca.png)

Mesmo processo, mas utilizando [t-SNE](https://www.kaggle.com/cast42/santander-customer-satisfaction/t-sne-manifold-visualisation/output):

![](https://github.com/gdmarmerola/santander-satisfaction/blob/master/tsne.png)

Foram dados nomes pouco explicativos para as variáveis, mas ainda era possível ter bons palpites sobre o que cada uma representava. Poucas tinham poder preditivo, sendo a mais importante a "var15" que muito provavelmente representa a idade do cliente.

![](https://github.com/gdmarmerola/santander-satisfaction/blob/master/var15.png)

Dessa forma, modelos simples conseguiram atingir resultados bastante próximos aos melhores na leaderboard. Um [script simples](https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3/comments) chegou ao top 10% da leaderboard pública. A diferença entre o meu resultado final (0.826928) e o resultado do vencedor da competição (0.829072) é de  0.002144 em termos de AUC. Muitas vezes, diferenças dessa ordem no resultado podem ser obtidas somente trocando a seed do gerador de números aleatórios do modelo. Logo, o principal desafio era garantir que a submissão enviada era estável no conjunto de validação local, na leaderboard pública e na privada. 

## 3. Competição

Nas competições Kaggle, existem duas leaderboards distintas que cumprem papéis diferentes nos rankings de usuários:

**Leaderboard pública:** contendo 50% dos dados de teste, essa é uma "amostra" da sua posição final na competição. Ela é constantemente atualizada e geralmente é um bom indicativo de performance.

**Leaderboard privada:** contém a outra parte dos 50% dos dados de teste. Só é revelada no final da competição e é sobre ela que se calcula o ranking final.

Uma das "táticas" usadas por alguns Kagglers é criar modelos locais utilizando os dados de treino e ajustar parâmetros de seus algoritmos conforme melhorias nos resultados da leaderboard pública. Nesta competição, em particular, houve uma onda de "scripts" que conseguiram resultados expressivos e de fato eram bastante difíceis de superar.

Porém, pela natureza ruidosa dos dados e pelo desequilíbrio da variável alvo, havia uma certa discrepância entre os dados da LB pública e privada. Em geral, aqueles que utilizaram a "tática" acima mencionada obtiveram uma surpresa ruim no resultado final. Alguns [fóruns](https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/20535/it-s-time-to-play-the-lb-shake-up-prediction-game?page=2) trataram esse tema, e de uma maneira geral, o **shake-up** (diferença entre as colocações nas LB's pública e privada) foi  muito grande: a maior queda foi de 3.147 posições, enquanto o maior ganho foi de 2,580 posições. Minha submissão final subiu 470 posições, de 940 na LB pública para 470	na LB privada. 

!["Obrigado pelo script". Indeed.](https://github.com/gdmarmerola/santander-satisfaction/blob/master/shakeup.png)

Isso mostra como ignorar a valiação local pode acarretar problemas em projetos que envolvam machine learning. Validar um modelo e ter confiança de que ele generaliza bem, é, sem dúvida, uma das maiores dificuldades da disciplina, mas também uma de suas maiores belezas. Nesse sentido, machine learning pode ser mesmo uma arte.

Foi um banho de água fria para uns, glória para outros. De qualquer forma, o maior aprendizado foi, [como mencionado](https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/20649/if-i-learned-something-today-was): *trust your CV*.

## 4. Modelagem

Desde o meu início no Kaggle eu faço forte uso de métodos de otimização de pipelines, algoritmos e hiperparâmetros (módulo `hyperopt`) e o processo de validação é um dos pilares para que tudo isso funcione bem. Um dos mitos do Kaggle, o brasileiro [Lucas Eustaquio](https://www.kaggle.com/leustagos) (que infelizmente faleceu um pouco antes do término da competição, por conta de câncer), já havia mostrado sua [preocupação](http://blog.kaggle.com/2016/02/22/profiling-top-kagglers-leustagos-current-7-highest-1/) com o processo de validação, mostrando que seu primeiro passo em qualquer competição é construir um conjunto de validação consistente.

Nesta competição, utilizei o `hyperopt` (TPE) para otimizar um pipeline de gradient boosting (XGBoost). Construí dois processos de validação:

**1o Round:** mais barato computacionalmente, utilizei uma CV estratificada de 10 folds (sem repetições) para ter uma visão geral dos hiperparâmetros e etapas da pipeline que mais importavam no problema.

**2o Round:** com um espaço de busca mais estreito obtido no passo anterior, usei uma custosa CV de 7-folds estratificada com 6 repetições, onde o melhor e pior resultado eram descartados.

Em geral, diversas técnicas foram utilizadas, mas o principal objetivo foi **estabilizar** as medições de AUC da validação local, uma vez que 0.003 na diferença de AUC no teste poderia significar uma queda de 1.350 posições na LB.

Centenas de modelos foram construídos pelo `hyperopt`, e minha submissão final foi um ensemble (média simples) de 7 pipelines diferentes. Todo o processo pode ser encontrado no notebook `model-selection`. Abaixo reproduzo uma vizualização interessante do processo de otimização de hiperparâmetros usado.

![](https://github.com/gdmarmerola/santander-satisfaction/blob/master/vis/hyperparameters/1st-round.png)

## 5. Conclusões

Além das lições sobre overfitting e validação, essa competição mostrou a importância da geração automática de modelos e otimização de hiperparâmetros. No futuro irei pesquisar mais sobre o tema e procurar implementar outras técnicas, como gaussian processes ou algoritmos genéticos. Espero que tenha sido uma boa leitura. Obrigado!!!
