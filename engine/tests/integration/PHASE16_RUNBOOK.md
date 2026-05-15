# Phase 16 — 48-hour Soak Runbook

This is the operator checklist for the final stability gate. The audit
machinery is automated; the soak itself is wall-clock and needs MT5 + the
Anthropic API + Discord wired up for real.

## Pre-flight

1. `.env` populated with real `MT5_*`, `ANTHROPIC_API_KEY`,
   `SUPABASE_*`, and at least `DISCORD_WEBHOOK_URL`. `FERNET_KEY`
   auto-generates on first launch.
2. Anthropic credit balance is non-zero (Claude gate fails
   `claude_unavailable` otherwise).
3. XM demo terminal is running and the 13 watchlist symbols are visible
   in Market Watch.
4. `scripts/install_engine.ps1` has been run as Administrator at least
   once so Task Scheduler can relaunch on reboot.
5. Truncate the prior soak's telemetry/log:
   ```powershell
   Remove-Item engine\logs\telemetry.jsonl -ErrorAction SilentlyContinue
   Remove-Item engine\logs\engine.log     -ErrorAction SilentlyContinue
   ```

## Run

1. Start the engine. Either launch the Task Scheduler job
   (`schtasks /Run /TN ComplexityEngine`) or run manually:
   ```powershell
   .\engine\.venv\Scripts\python.exe -m engine.engine
   ```
2. Open the UI (`pnpm --dir ui run electron:dev` while iterating, or the
   packaged build in production).
3. Note the soak start timestamp (UTC) — the audit window begins here.
4. Leave the system idle for 48 hours. Telemetry rows land every 30s
   in `engine/logs/telemetry.jsonl`; notification counters are bumped
   by `windows_toast.notify()` straight into `settings_kv`.

## Trigger every notification path

The audit fails unless every one of the 8 notification events fires at
least once during the window. Most fire naturally during normal trading,
but two are unlikely to surface organically:

* `KILL_TRIGGERED` — fire `cmd_emergency_close` from the UI Emergency
  Stop button (closes any open positions and records the toast).
* `TRAINING_COMPLETE` — trigger a manual retrain from
  AI Engine → "Retrain" → Confirm; the worker writes a new checkpoint
  and the daily summary path emits the toast.

## Audit

After the 48 hours, run:

```powershell
.\engine\.venv\Scripts\python.exe -m engine.utils.telemetry audit --hours 48
```

Exit code `0` = PASS, `2` = FAIL. The output lists each criterion with
`[PASS]` / `[FAIL]` markers and observed values (RSS p99, CPU p95, trades,
unhandled exceptions, missing notifications, Supabase row delta).

To re-run the audit on a custom window or skip Supabase comparison:

```powershell
python -m engine.utils.telemetry audit --hours 24 --no-supabase
```

## Pass thresholds (mirrors §15 Phase 16)

| Criterion              | Threshold                             |
|------------------------|---------------------------------------|
| trades_executed        | ≥ 6 (3/day × 2 days)                  |
| memory p99             | ≤ 600 MB                              |
| process CPU p95        | ≤ 25 %                                |
| system CPU p95         | ≤ 5 % (or proc CPU under cap)         |
| unhandled exceptions   | 0 (matched against engine.log)        |
| notification coverage  | all 8 event types observed            |
| Supabase row count     | matches local SQLite trade count       |
| telemetry samples      | ≥ 1 sample present in window          |

## CI gate

`engine/tests/integration/test_48hr_report.py` exercises the audit
function against synthetic fixtures — both the PASS path and one
failure variant per criterion — so changes to thresholds or the audit
logic are caught without paying for a real 48-hour wall-clock run.
