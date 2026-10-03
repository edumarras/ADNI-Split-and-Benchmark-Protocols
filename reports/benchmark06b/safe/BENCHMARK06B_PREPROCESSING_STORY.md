# BENCHMARK06B — Pré-processamento das entradas

- Versão do script: `0.1.0`
- Política: `2.0.0-crosschecked-b`
- Fit: treino somente
- Validação reaprendida: **não**
- Teste usado/materializado: **não**
- Targets MRI carregados: **não**

## Política

- transformações semânticas e suporte são consumidos dos locks anteriores;
- numéricos usam mediana, standardization e indicador de ausência;
- binários usam mediana, sem scaling, e indicador de ausência;
- categóricos usam `__MISSING__` e one-hot com unknown ignorado;
- multi-response usa vocabulário do treino e multi-hot;
- blocos históricos temporalmente inseguros são anulados antes de qualquer transformação;
- cada fonte/bloco recebe `modality_present` após essa anulação e antes da imputação;
- variância zero é removida somente pelo treino.

## Dimensões

| Cenário | Treino | Validação | Originais | Antes var. | Finais | Densidade treino |
|---|---:|---:|---:|---:|---:|---:|
| `snapshot_all` | 6707 | 1445 | 220 | 667 | 660 | 0.4358 |
| `snapshot_matched_last` | 1419 | 299 | 219 | 663 | 656 | 0.3854 |
| `longitudinal_previous_last` | 1419 | 299 | 427 | 1260 | 1248 | 0.4029 |

## Unknowns na validação

Unknowns são contabilizados, mas valores/tokens não são escritos nos relatórios safe.

| Cenário | Tipo | Features afetadas | Ocorrências |
|---|---|---:|---:|
| `snapshot_all` | `categorical_value` | 1 | 1 |
| `snapshot_all` | `multi_response_token` | 0 | 0 |
| `snapshot_matched_last` | `categorical_value` | 2 | 3 |
| `snapshot_matched_last` | `multi_response_token` | 0 | 0 |
| `longitudinal_previous_last` | `categorical_value` | 12 | 14 |
| `longitudinal_previous_last` | `multi_response_token` | 0 | 0 |

## Variância zero

| Cenário | Antes | Removidas | Finais |
|---|---:|---:|---:|
| `snapshot_all` | 667 | 7 | 660 |
| `snapshot_matched_last` | 663 | 7 | 656 |
| `longitudinal_previous_last` | 1260 | 12 | 1248 |

## Artefatos

Matrizes, preprocessadores, row manifests, medianas, categorias e tokens foram escritos somente em `data/processed/local_only/benchmark06b_preprocessed_inputs/`.
Essa pasta é individualizada/local-only e não pode ser compartilhada.
