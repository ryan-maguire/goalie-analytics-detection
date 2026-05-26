# metrics_seg diff: v13 vs v14_prefilter

**Video:** `SX5xNJlh6eQ`
**Segments compared:** 59
**Segments where output differed:** 29
**GT events total in this video:** 71 (3 goals)

## Verdict tally (only counting segments that differed)

| field | v13 closer to GT | v14 closer to GT | tied |
|---|---|---|---|
| goals | 1 | 0 | 28 |
| shotsOnNet | 16 | 8 | 5 |

## Per-segment differences

### Segment 01:03–01:33 (63-93s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 1 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 1 | v13_closer |
| saves | 2 | 4 | +2 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 01:26-01:38 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 01:34–02:05 (94-125s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 1 | — | 1 | — |
| shotsOnNet | 1 | 1 | — | 1 | — |
| saves | 0 | 1 | +1 | 0 | v13_closer |
| goals | 1 | 0 | -1 | 1 | v13_closer |

**GT events in window:**
- 01:40-01:52 `Goals` `Philadelphia Jr. Flyers 19U AA`
- 01:40-01:52 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=3; per_call_goals=[1, 1, 1]; goal_vote=kept
- v14 trace: n_calls=3; per_call_goals=[1, 0]; goal_vote=rejected

- v13 per-call goal counts: [1, 1, 1]
- v14 per-call goal counts: [1, 0]

---

### Segment 03:38–04:08 (218-248s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 4 | +1 | 0 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 0 | v13_closer |
| saves | 2 | 4 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 04:40–05:10 (280-310s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 08:48–09:18 (528-558s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 0 | v13_closer |
| shotsOnNet | 1 | 2 | +1 | 0 | v13_closer |
| saves | 1 | 2 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 09:50–10:20 (590-620s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 3 | — | 2 | — |
| shotsOnNet | 2 | 3 | +1 | 2 | v13_closer |
| saves | 2 | 3 | +1 | 2 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 10:00-10:12 `Shots` `Philadelphia Jr. Flyers 19U AA`
- 10:14-10:26 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 10:52–11:22 (652-682s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 3 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=3; per_call_goals=[1, 0, 0]; goal_vote=rejected
- v14 trace: n_calls=1

---

### Segment 16:10–16:41 (970-1001s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 1 | v13_closer |
| shotsOnNet | 1 | 1 | — | 1 | — |
| saves | 1 | 1 | — | 1 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 16:21-16:33 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 18:35–19:35 (1115-1175s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 1 | -1 | 1 | v14_closer |
| shotsOnNet | 2 | 1 | -1 | 1 | v14_closer |
| saves | 2 | 1 | -1 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 19:23-19:35 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 20:39–21:09 (1239-1269s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 24:55–25:25 (1495-1525s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 3 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 25:57–26:27 (1557-1587s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 5 | +3 | 1 | v13_closer |
| shotsOnNet | 2 | 5 | +3 | 1 | v13_closer |
| saves | 2 | 5 | +3 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 26:03-26:15 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied

---

### Segment 29:28–30:18 (1768-1818s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 30:19–30:49 (1819-1849s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 0 | -4 | 1 | v14_closer |
| shotsOnNet | 4 | 0 | -4 | 1 | v14_closer |
| saves | 4 | 0 | -4 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 30:19-30:31 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied
- v14 trace: n_calls=1; FAIL=first_call_failed

---

### Segment 30:50–31:42 (1850-1902s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 0 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 0 | v13_closer |
| saves | 2 | 4 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied

---

### Segment 32:23–32:53 (1943-1973s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 0 | v13_closer |
| shotsOnNet | 1 | 2 | +1 | 0 | v13_closer |
| saves | 1 | 2 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 34:51–35:36 (2091-2136s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 1 | v13_closer |
| shotsOnNet | 0 | 0 | — | 1 | — |
| saves | 0 | 0 | — | 1 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 35:22-35:34 `Shots` `North Shore Warhawks 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 35:44–36:14 (2144-2174s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 0 | v13_closer |
| shotsOnNet | 2 | 3 | +1 | 0 | v13_closer |
| saves | 2 | 3 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 41:25–41:45 (2485-2505s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 1 | v14_closer |
| shotsOnNet | 3 | 1 | -2 | 1 | v14_closer |
| saves | 3 | 1 | -2 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 41:28-41:40 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 43:03–43:33 (2583-2613s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 3 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 45:38–46:08 (2738-2768s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 1 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 1 | v14_closer |
| saves | 3 | 2 | -1 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 45:54-46:06 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 46:24–46:55 (2784-2815s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 3 | +1 | 1 | v13_closer |
| saves | 2 | 3 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 46:35-46:47 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 47:34–48:02 (2854-2882s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 0 | -2 | 0 | v14_closer |
| shotsOnNet | 2 | 0 | -2 | 0 | v14_closer |
| saves | 2 | 0 | -2 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 49:44–50:14 (2984-3014s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 0 | -3 | 2 | v13_closer |
| shotsOnNet | 3 | 0 | -3 | 2 | v13_closer |
| saves | 3 | 0 | -3 | 2 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 49:42-49:54 `Shots` `Philadelphia Jr. Flyers 19U AA`
- 49:48-50:00 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1; FAIL=first_call_failed

---

### Segment 51:22–51:52 (3082-3112s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 1 | -2 | 2 | tied_off |
| shotsOnNet | 3 | 1 | -2 | 2 | tied_off |
| saves | 3 | 1 | -2 | 2 | tied_off |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 51:30-51:42 `Shots` `Philadelphia Jr. Flyers 19U AA`
- 51:31-51:43 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied
- v14 trace: n_calls=1

---

### Segment 52:57–53:27 (3177-3207s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 53:28–53:58 (3208-3238s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 3 | +1 | 1 | v13_closer |
| saves | 2 | 3 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 53:52-54:04 `Shots` `Philadelphia Jr. Flyers 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 54:27–55:08 (3267-3308s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 0 | 2 | +2 | 0 | v13_closer |
| shotsOnNet | 0 | 2 | +2 | 0 | v13_closer |
| saves | 0 | 2 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 58:18–58:48 (3498-3528s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 1 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 1 | — | 0 | — |
| saves | 1 | 1 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---
