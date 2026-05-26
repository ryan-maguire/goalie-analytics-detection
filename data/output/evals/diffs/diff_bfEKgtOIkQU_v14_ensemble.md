# metrics_seg diff: v13 vs v14_ensemble

**Video:** `bfEKgtOIkQU`
**Segments compared:** 81
**Segments where output differed:** 34
**GT events total in this video:** 68 (2 goals)

## Verdict tally (only counting segments that differed)

| field | v13 closer to GT | v14 closer to GT | tied |
|---|---|---|---|
| goals | 0 | 0 | 34 |
| shotsOnNet | 14 | 12 | 8 |

## Per-segment differences

### Segment 00:45–01:15 (45-75s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 2 | — | 1 | — |
| shotsOnNet | 1 | 2 | +1 | 1 | v13_closer |
| saves | 1 | 2 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 01:03-01:15 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 01:47–02:17 (107-137s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 2 | 0 | -2 | 0 | v14_closer |
| saves | 2 | 0 | -2 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 02:19–02:49 (139-169s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 5 | +1 | 0 | v13_closer |
| shotsOnNet | 3 | 4 | +1 | 0 | v13_closer |
| saves | 3 | 4 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied

---

### Segment 06:15–06:41 (375-401s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 1 | v13_closer |
| shotsOnNet | 1 | 2 | +1 | 1 | v13_closer |
| saves | 1 | 2 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 06:18-06:30 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 06:49–07:19 (409-439s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 0 | 2 | +2 | 1 | tied_off |
| shotsOnNet | 0 | 2 | +2 | 1 | tied_off |
| saves | 0 | 2 | +2 | 1 | tied_off |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 07:04-07:16 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1; FAIL=first_call_failed
- v14 trace: n_calls=1

---

### Segment 08:41–09:11 (521-551s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 10:29–10:58 (629-658s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 2 | — | 1 | — |
| saves | 2 | 2 | — | 1 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 10:40-10:52 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 11:33–12:03 (693-723s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 12:04–12:34 (724-754s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 0 | v13_closer |
| shotsOnNet | 1 | 1 | — | 0 | — |
| saves | 1 | 1 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 12:46–13:16 (766-796s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 4 | +1 | 1 | v13_closer |
| shotsOnNet | 2 | 2 | — | 1 | — |
| saves | 2 | 2 | — | 1 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 13:06-13:18 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied
- v14 trace: n_calls=1

---

### Segment 17:27–17:57 (1047-1077s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 3 | +2 | 0 | v13_closer |
| shotsOnNet | 1 | 3 | +2 | 0 | v13_closer |
| saves | 1 | 3 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 20:02–20:32 (1202-1232s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 21:38–21:54 (1298-1314s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 4 | +2 | 0 | v13_closer |
| shotsOnNet | 2 | 4 | +2 | 0 | v13_closer |
| saves | 2 | 4 | +2 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 24:36–25:06 (1476-1506s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 2 | v13_closer |
| shotsOnNet | 2 | 2 | — | 2 | — |
| saves | 2 | 2 | — | 2 | — |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 24:40-24:52 `Shots` `Chicago Hawks 19U`
- 24:58-25:10 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 27:50–28:20 (1670-1700s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 2 | -2 | 0 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 28:21–28:51 (1701-1731s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 0 | v13_closer |
| shotsOnNet | 1 | 2 | +1 | 0 | v13_closer |
| saves | 1 | 2 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 28:52–29:13 (1732-1753s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 2 | +1 | 1 | v13_closer |
| shotsOnNet | 1 | 2 | +1 | 1 | v13_closer |
| saves | 1 | 2 | +1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 28:51-29:03 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 32:20–32:50 (1940-1970s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 37:54–38:24 (2274-2304s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 0 | 1 | +1 | 0 | v13_closer |
| shotsOnNet | 0 | 1 | +1 | 0 | v13_closer |
| saves | 0 | 1 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 38:25–38:55 (2305-2335s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 3 | — | 0 | — |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 40:13–40:43 (2413-2443s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 1 | v13_closer |
| shotsOnNet | 1 | 0 | -1 | 1 | v13_closer |
| saves | 1 | 0 | -1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 40:28-40:40 `Shots` `North Shore Warhawks 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 40:58–41:28 (2458-2488s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 5 | +4 | 0 | v13_closer |
| shotsOnNet | 1 | 5 | +4 | 0 | v13_closer |
| saves | 1 | 5 | +4 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied
- v14 trace: n_calls=3; per_call_goals=[0, 0, 0]; shot_vote=applied

---

### Segment 43:57–44:27 (2637-2667s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 1 | 0 | -1 | 0 | v14_closer |
| saves | 1 | 0 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1; FAIL=first_call_failed

---

### Segment 46:19–46:44 (2779-2804s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 0 | -2 | 1 | tied_off |
| shotsOnNet | 2 | 0 | -2 | 1 | tied_off |
| saves | 2 | 0 | -2 | 1 | tied_off |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 46:21-46:33 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1; FAIL=first_call_failed

---

### Segment 50:09–50:39 (3009-3039s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 0 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 51:20–51:50 (3080-3110s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 3 | +1 | 0 | v13_closer |
| shotsOnNet | 2 | 2 | — | 0 | — |
| saves | 2 | 2 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 52:22–52:52 (3142-3172s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 2 | 1 | -1 | 1 | v14_closer |
| shotsOnNet | 2 | 1 | -1 | 1 | v14_closer |
| saves | 2 | 1 | -1 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 52:16-52:28 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 52:53–53:23 (3173-3203s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 0 | v14_closer |
| shotsOnNet | 0 | 0 | — | 0 | — |
| saves | 0 | 0 | — | 0 | — |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 58:05–58:35 (3485-3515s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 1 | 0 | -1 | 1 | v13_closer |
| shotsOnNet | 1 | 0 | -1 | 1 | v13_closer |
| saves | 1 | 0 | -1 | 1 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 58:24-58:36 `Shots` `North Shore Warhawks 19U AA`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 61:38–62:08 (3698-3728s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 0 | 1 | +1 | 0 | v13_closer |
| shotsOnNet | 0 | 1 | +1 | 0 | v13_closer |
| saves | 0 | 1 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 62:41–63:11 (3761-3791s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 3 | -1 | 0 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 0 | v14_closer |
| saves | 3 | 2 | -1 | 0 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 65:02–65:32 (3902-3932s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 4 | 3 | -1 | 1 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 1 | v14_closer |
| saves | 3 | 2 | -1 | 1 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 65:07-65:19 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 66:04–66:34 (3964-3994s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 2 | -1 | 2 | v14_closer |
| shotsOnNet | 3 | 2 | -1 | 2 | v14_closer |
| saves | 3 | 2 | -1 | 2 | v14_closer |
| goals | 0 | 0 | — | 0 | — |

**GT events in window:**
- 66:00-66:12 `Shots` `North Shore Warhawks 19U AA`
- 66:15-66:27 `Shots` `Chicago Hawks 19U`

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---

### Segment 68:20–68:50 (4100-4130s)

| field | v13 | v14 | Δ | GT (in window) | verdict |
|---|---|---|---|---|---|
| shots | 3 | 4 | +1 | 0 | v13_closer |
| shotsOnNet | 3 | 4 | +1 | 0 | v13_closer |
| saves | 3 | 4 | +1 | 0 | v13_closer |
| goals | 0 | 0 | — | 0 | — |

- v13 trace: n_calls=1
- v14 trace: n_calls=1

---
