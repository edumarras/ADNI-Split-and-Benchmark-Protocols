# SPLIT06 — Auditoria de visitas anteriores na mesma onda protocolar

- Versão do script: `0.1.0`
- Escopo: treino e validação somente
- Teste usado: **não**
- Pacote congelado modificado: **não**
- Dados individualizados escritos: **não**

## Pergunta auditada

`sc`, `scmri`, `bl`, `m00` e `v01` recebem mês protocolar zero, mas
continuam sendo chaves exatas `RID + VISCODE2` distintas. Quando mais de
uma dessas visitas aparece na onda anterior mais recente, o benchmark
precisa escolher uma visita real ou excluir o participante; concatená-las
criaria uma visita sintética.

## Reprodução do problema

- Treino: **269** participantes ambíguos.
- Validação: **58** participantes ambíguos.

## Políticas comparadas

1. Intervalos de data estritos.
2. Consenso exato de data.
3. Mediana das datas apenas como sensibilidade.
4. Ordem explícita `sc < scmri < bl < m00` apenas como sensibilidade.
5. União hipotética apenas para medir ganho; visitas nunca são concatenadas.

## Validação

- Checks executados: **9**
- Checks falhos: **0**
