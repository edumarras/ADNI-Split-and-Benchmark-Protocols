# BENCHMARK05B — Construção auditada dos cenários

- Status esperado após execução: **PASS**
- Dados utilizados: **treino e validação somente**
- Teste materializado: **não**
- Modelos treinados: **não**
- Threshold congelado: **50 participantes distintos do fit**
- Participantes pareados no treino: **1419**
- Participantes pareados na validação: **299**

## Cenários

### `snapshot_all`

Inclui todas as visitas target-complete de treino e validação. Cada linha recebe
somente as entradas não-MRI da própria visita, além de PTDEMOG e APOERES por
participante.

### `snapshot_matched_last`

Seleciona a última visita target-complete de cada participante segundo a ordem
congelada do pacote. O participante só entra se essa visita atual possuir uma
visita não-MRI anterior elegível. Não existe fallback para uma visita-alvo mais
antiga quando a última visita não pode ser pareada.

### `longitudinal_previous_last`

Usa exatamente os mesmos participantes, visitas atuais e targets de
`snapshot_matched_last`. Acrescenta uma única visita anterior comum a todas as
modalidades, consumindo o VISCODE2 exato congelado pelo `SPLIT07`. Quando havia
mais de uma candidata na mesma onda, somente intervalos clínicos estritamente
ordenados puderam desempatar.

A visita anterior não precisa possuir MRI nem target completo. Ela precisa ter
pelo menos uma feature visit-level semanticamente autorizada observada. Não há
carry-forward e não se busca uma visita histórica diferente para cada modalidade.
Blocos de uma fonte anterior cuja data clínica ocorreu no mesmo dia ou depois da
MRI atual são representados como ausentes antes do cálculo de suporte. Um par só é
mantido se ainda restar alguma entrada anterior temporalmente segura.

## Attrition do pareamento

| Split | Status | Motivo | Participantes |
|---|---|---|---:|
| train | eligible | `eligible_strict_date_interval_winner` | 256 |
| train | eligible | `eligible_unique_previous_visit` | 1163 |
| train | excluded | `ambiguous_same_wave_no_strict_interval_winner` | 9 |
| train | excluded | `no_semantically_authorized_previous_input_after_crosscheck` | 7 |
| train | excluded | `only_unknown_or_non_earlier_history` | 548 |
| train | excluded | `strict_latest_not_fully_before_current` | 4 |
| validation | eligible | `eligible_strict_date_interval_winner` | 55 |
| validation | eligible | `eligible_unique_previous_visit` | 244 |
| validation | excluded | `ambiguous_same_wave_no_strict_interval_winner` | 2 |
| validation | excluded | `no_semantically_authorized_previous_input_after_crosscheck` | 2 |
| validation | excluded | `only_unknown_or_non_earlier_history` | 122 |
| validation | excluded | `strict_latest_not_fully_before_current` | 1 |

## Intervalo temporal

| Split | Pares | Mediana (meses) | Média | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| train | 1419 | 12.00 | 14.39 | 3.00 | 72.00 |
| validation | 299 | 12.00 | 14.23 | 3.00 | 60.00 |

## Suporte por cenário e bloco

A regra de 50 é recalculada usando somente participantes do treino. PTDEMOG e
APOERES formam o bloco `participant_static` e aparecem uma única vez. No cenário
longitudinal, `current_visit` e `previous_visit` têm suporte calculado
separadamente.

| Cenário | Bloco | Feature instances retidas |
|---|---|---:|
| `longitudinal_previous_last` | `current_derived` | 1 |
| `longitudinal_previous_last` | `current_visit` | 200 |
| `longitudinal_previous_last` | `participant_static` | 10 |
| `longitudinal_previous_last` | `previous_visit` | 201 |
| `snapshot_all` | `current_derived` | 1 |
| `snapshot_all` | `current_visit` | 201 |
| `snapshot_all` | `participant_static` | 10 |
| `snapshot_matched_last` | `current_derived` | 1 |
| `snapshot_matched_last` | `current_visit` | 200 |
| `snapshot_matched_last` | `participant_static` | 10 |

## Decisões conservadoras

- códigos de visita sem mês protocolar interpretável não são usados como histórico;
- `sc` e `bl` permanecem visitas exatas distintas e nunca são concatenadas;
- empates na mesma onda são resolvidos apenas por intervalos de datas estritos;
- casos ainda inconclusivos são excluídos, sem mediana, ordem lexical ou hierarquia;
- a população do cenário estático pareado é idêntica à longitudinal;
- o teste permanece fechado e será materializado somente após o lock final dos modelos.

## Artefatos local-only

Os manifests com RID/VISCODE2 foram escritos em
`data/processed/local_only/benchmark05b_scenarios/`. Eles não podem ser enviados,
publicados ou commitados.

Hashes globais dos manifests locais:

```json
{
  "snapshot_all_development_anchor.csv.gz": "08c372239ad46e396d04b3757576280e0efc9148be54e0747f3ab7afd8eab8da",
  "snapshot_matched_last_development_anchor.csv.gz": "b0738f6e4bb2bd30dcdba8ca28ad3e59c477f9be0a86055b715ef868692169dc",
  "longitudinal_previous_last_development_pairs.csv.gz": "92596f56081e8309c613eb519c67f0ae09bd676769ff7b936cf819cf9edebc00",
  "development_visit_availability_timeline.csv.gz": "e66ad844fd846a76982581c791dd5d2e211f02074bef112f40e47f43fae66aed",
  "pairing_exclusions.csv.gz": "e6fe2942cae1b2be2ae58a091d7e7940ef39c7d62d7e9ef8a13909e00251c5a3",
  "LOCAL_ONLY_age_at_target_derivation.csv.gz": "2c28184c4994cda63849f247328337a118887fff05fa26a543d13f9a7ceedbe3"
}
```
