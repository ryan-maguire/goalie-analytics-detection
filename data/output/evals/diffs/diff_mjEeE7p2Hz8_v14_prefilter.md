# metrics_seg diff: v13 vs v14_prefilter

**Video:** `mjEeE7p2Hz8`
**Segments compared:** 71
**Segments where output differed:** 24
**GT events total in this video:** 76 (7 goals)

## Verdict tally (only counting segments that differed)

| field | v13 closer to GT | v14 closer to GT | tied |
|---|---|---|---|
| goals | 1 | 2 | 21 |
| shotsOnNet | 9 | 10 | 5 |

## Per-segment differences

### Segment 03:11–03:41 (191-221s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 2 | -2 | 2 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 2 | v14_closer |
| saves | 3 | 2 | -1 | 2 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 03:12-03:24 `Shots` `Amherst Lady Knights 19U`
- 03:33-03:45 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 07:50–08:20 (470-500s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 2 | -2 | 1 | v14_closer |
| shotsOnNet | 4 | 2 | -2 | 1 | v14_closer |
| saves | 4 | 2 | -2 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 07:57-08:09 `Shots` `North Shore Warhawks 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 10:25–10:55 (625-655s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 1 | — | 1 | — |
| shotsOnNet | 1 | 1 | — | 1 | — |
| saves | 0 | 1 | +1 | 0 | v13_closer |
| goals | 1 | 0 | -1 | 1 | v13_closer |

**GT events in window:**
- 10:29-10:41 `Goals` `Amherst Lady Knights 19U`
- 10:29-10:41 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=3; per_call_goals=[1, 1, 1]; goal_vote=kept
- v14 trace: n_calls=1

- v13 per-call goal counts: [1, 1, 1]
- v14 per-call goal counts: [0]

---

### Segment 10:56–11:26 (656-686s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 3 | +1 | 1 | v13_closer |
| saves | 2 | 3 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 11:11-11:23 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 11:58–12:28 (718-748s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 2 | 1 | -1 | 0 | v14_closer |
| saves | 2 | 1 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 14:21–14:51 (861-891s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 0 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 0 | v13_closer |
| saves | 2 | 4 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 14:52–15:22 (892-922s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 2 | — | 0 | — |
| shotsOnNet | 2 | 1 | -1 | 0 | v14_closer |
| saves | 2 | 1 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 20:35–21:05 (1235-1265s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 1 | — | 1 | — |
| shotsOnNet | 1 | 1 | — | 1 | — |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 1 | +1 | 1 | v14_closer |

**GT events in window:**
- 20:40-20:52 `Goals` `Amherst Lady Knights 19U`
- 20:40-20:52 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=3; per_call_goals=[1, 0, 0]; goal_vote=rejected
- v14 trace: n_calls=3; per_call_goals=[1, 0, 1]; goal_vote=kept

- v13 per-call goal counts: [1, 0, 0]
- v14 per-call goal counts: [1, 0, 1]

---

### Segment 22:08–22:38 (1328-1358s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 1 | -1 | 1 | v14_closer |
| shotsOnNet | 1 | 1 | — | 1 | — |
| saves | 1 | 1 | — | 1 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 22:15-22:27 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 24:12–24:42 (1452-1482s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 25:14–25:44 (1514-1544s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 2 | — | 1 | — |
| shotsOnNet | 1 | 2 | +1 | 1 | v13_closer |
| saves | 1 | 2 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 25:31-25:43 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 25:45–26:15 (1545-1575s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 2 | 2 | — | 0 | — |
| saves | 2 | 2 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 26:47–27:17 (1607-1637s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 1 | — | 0 | — |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 27:49–28:19 (1669-1699s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 0 | 0 | — | 0 | — |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 0 | 0 | — | 0 | — |
| goals | 1 | 0 | -1 | 0 | v14_closer |

- v13 trace: n_calls=3; per_call_goals=[1, 0, 1]; goal_vote=kept
- v14 trace: n_calls=1

- v13 per-call goal counts: [1, 0, 1]
- v14 per-call goal counts: [0]

---

### Segment 33:04–33:33 (1984-2013s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 3 | +1 | 1 | v13_closer |
| saves | 2 | 3 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 33:03-33:15 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 33:35–33:59 (2015-2039s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 2 | — | 1 | — |
| shotsOnNet | 1 | 2 | +1 | 1 | v13_closer |
| saves | 1 | 2 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 33:49-34:01 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 37:49–38:19 (2269-2299s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 1 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 1 | v13_closer |
| saves | 2 | 4 | +2 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 38:10-38:22 `Shots` `North Shore Warhawks 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 39:22–40:14 (2362-2414s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 1 | -3 | 0 | v14_closer |
| shotsOnNet | 4 | 1 | -3 | 0 | v14_closer |
| saves | 4 | 1 | -3 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 40:55–41:55 (2455-2515s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 3 | — | 0 | — |
| shotsOnNet | 2 | 3 | +1 | 0 | v13_closer |
| saves | 2 | 3 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 41:57–42:27 (2517-2547s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 3 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=3; per_call_goals=[1, 0, 0]; goal_vote=rejected
- v14 trace: n_calls=1

---

### Segment 46:15–46:45 (2775-2805s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 5 | 4 | -1 | 0 | v14_closer |
| shotsOnNet | 4 | 3 | -1 | 0 | v14_closer |
| saves | 4 | 3 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied
- v14 trace: n_calls=1

---

### Segment 47:17–47:47 (2837-2867s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 1 | — | 0 | — |
| shotsOnNet | 0 | 1 | +1 | 0 | v13_closer |
| saves | 0 | 1 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 47:48–48:18 (2868-2898s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 1 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 1 | v14_closer |
| saves | 3 | 2 | -1 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 47:44-47:56 `Shots` `Amherst Lady Knights 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 52:02–52:32 (3122-3152s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 1 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 1 | — | 0 | — |
| saves | 1 | 1 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---
