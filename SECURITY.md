# Security Policy

## Supported versions

Forex AI is experimental software. Security updates are applied to the
latest `main` branch only. There are no backport releases.

| Version | Supported |
| --- | --- |
| `main` (latest) | Yes |
| Tagged releases | Best-effort, no SLA |
| Anything else | No |

## Reporting a vulnerability

**Do NOT open a public GitHub issue for security bugs.**

Email the maintainer directly. Include:

1. Description of the issue and its impact (preferably with a proof of
   concept).
2. Affected version / commit SHA.
3. Suggested fix, if you have one.
4. Whether you'd like credit in the disclosure.

You should receive an initial response within 72 hours. If the issue is
confirmed, we'll work on a fix and coordinate a disclosure date with you
(typically within 30 days).

## Disclosure policy

- We acknowledge receipt within 72 hours.
- We investigate and confirm/deny within 7 days.
- We develop a fix and target a release within 30 days (sooner for
  critical issues).
- We publish a GitHub Security Advisory with credits (unless you prefer
  to remain anonymous).
- We do NOT publicly disclose until a fix is available, except in cases
  of active exploitation.

## Scope

In scope:

- Anything in this repository that allows an attacker to:
  - Place unauthorized trades
  - Read or extract secrets (MT5 credentials, Telegram token, LLM API keys)
  - Bypass risk controls (kill switch, drawdown limits, human override)
  - Crash the trading loop
  - Inject malicious analysis data

Out of scope:

- Vulnerabilities in third-party dependencies (report to upstream)
- Vulnerabilities in MT5 terminal itself (report to MetaQuotes)
- Vulnerabilities in Telegram's infrastructure (report to Telegram)
- Social engineering / phishing against the maintainer
- Issues that require physical access to the host

## Security measures in this project

See [`docs/security.md`](docs/security.md) for the full threat model and
the security measures already in place. Highlights:

- `.env` gitignored, never baked into Docker images (`.dockerignore`)
- MT5 password held in memory only, never logged
- Paper mode by default; live MT5 requires explicit `mt5_live`
- Human override (STOP_ALL, CLOSE_ALL, PAUSE, RESUME) via Telegram or file
- Trading-as-Git approval gate for live trades (optional, recommended)
- Kill switch on daily drawdown threshold
- Circuit breaker on consecutive losing trades
- Magic number isolation from manual trades in the same MT5 terminal
- `pip-audit` in CI

## Hardening recommendations for operators

If you're running Forex AI in production:

1. **Host**: dedicated VPS with full-disk encryption (LUKS / BitLocker).
2. **Network**: firewall egress to only MT5 server IPs, Telegram API,
   and your LLM provider. Block all inbound except SSH.
3. **SSH**: key-only auth, fail2ban, non-root user.
4. **MT5 account**: dedicated trading account, not your main account.
   Start with a demo account; promote to live only after 30 days of
   stable paper + demo operation.
5. **Telegram bot**: dedicated bot, restricted to `ALLOWED_USER_IDS`.
6. **Backups**: encrypted offsite, rotation 30 days.
7. **Monitoring**: enable Prometheus + Grafana (see
   `monitoring/prometheus.yml`); alert on kill switch events.
8. **Trading-as-Git**: enable (`JOURNAL_ENABLED=true`) for any live
   trading. Review staged/committed intentions daily.

## Disclaimer

This software is provided "as is", without warranty of any kind. The
authors provide no guarantees of correctness, reliability, profitability,
or loss prevention. Trading forex carries substantial risk; you can lose
some or all of your invested capital. Use at your own risk.
