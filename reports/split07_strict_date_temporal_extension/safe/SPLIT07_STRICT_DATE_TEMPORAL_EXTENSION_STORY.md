# SPLIT07 — Congelamento da política temporal por datas estritas

- Versão: `0.1.0`
- Política: `strict_date_interval_same_wave_previous_visit_v1`
- Decision ID histórico preservado: `90fefe5ec34d49743612`
- Split oficial alterado: **não**
- População supervisionada alterada: **não**
- Targets MRI alterados: **não**
- Teste usado: **não**
- Visitas concatenadas: **não**

## Regra congelada

Quando mais de uma visita exata compete no mês protocolar anterior mais recente, uma candidata só é selecionada quando sua menor data clínica é posterior à maior data clínica de todas as concorrentes.
A maior data da visita anterior também precisa ser estritamente anterior à menor data da visita-alvo atual.

Não há fallback por mediana, ordem lexical ou hierarquia de VISCODE2.

## Resultado em desenvolvimento

Train: 1170 únicos + 256 recuperados = 1426.
Validation: 246 únicos + 55 recuperados = 301.

## Reconstrução

O identificador temporal histórico foi preservado. O payload reconstruído recebe um SHA-256 próprio porque o ID original dependia do hash de um resumo SPLIT06 com timestamp de execução.

## Validação

- Checks executados: **27**
- Checks falhos: **0**
