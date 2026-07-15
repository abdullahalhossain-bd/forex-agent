# server/webhook_server.py  —  Day 32 | TradingView Webhook Receiver
# ============================================================
# TradingView Pine Script alert → এখানে POST আসে → existing
# pipeline (rule engine + AIAnalyst + DecisionAgent + CircuitBreaker
# + ExecutionRouter) চালায়।
#
# এটা TradingView automation/scraping না — এটা TradingView-এর
# অফিসিয়াল, supported "Webhook URL" alert feature ব্যবহার করে।
# কোনো browser click simulate করা হচ্ছে না, ToS-এর মধ্যেই থাকছে।
#
# Flow:
#   Pine Script alert
#       ↓ POST JSON
#   /webhook/tradingview  (এই ফাইল)
#       ↓ secret verify
#   SignalPipeline.process()
#       ↓
#   rule engine + LLM → DecisionAgent → CircuitBreaker → ExecutionRouter
# ============================================================

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("webhook_server")

app = Flask(__name__)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

if not WEBHOOK_SECRET:
    log.warning(
        "WEBHOOK_SECRET সেট করা নেই .env-এ — কেউ চাইলে ভুয়া signal পাঠাতে পারবে। "
        "Production-এ এটা mandatory করা উচিত।"
    )


@app.route("/webhook/tradingview", methods=["POST"])
def tradingview_webhook():
    """
    TradingView alert এখানে POST করে।

    Pine Script alert setup করার সময় "Webhook URL" ফিল্ডে এই endpoint-এর
    full URL দিতে হবে (যেমন https://yourserver.com/webhook/tradingview)।
    Message body-তে ai_trader_webhook.pine-এর jsonMsg variable টা ব্যবহার হয়।

    Security note: TradingView webhook নিজে কোনো auth header পাঠাতে পারে না,
    তাই secret-টা payload-এর ভেতরেই embed করতে হয় (alert message-এর শুরুতে),
    অথবা URL-এর query param হিসেবে (?secret=xxx) — দ্বিতীয়টা সহজ এবং এখানে
    সেটাই ব্যবহার করা হয়েছে।
    """
    secret = request.args.get("secret", "")
    # Day 102+ CRITICAL hotfix: FAIL CLOSED when WEBHOOK_SECRET is unset.
    # Previously: `if WEBHOOK_SECRET and secret != WEBHOOK_SECRET` — when
    # the env var was unset (default ""), the auth check was skipped
    # entirely, allowing anyone with network access to port 5000 to POST
    # a trade signal and trigger a real MT5 order via SignalPipeline →
    # ExecutionRouter. A missing secret must NOT silently disable auth.
    if not WEBHOOK_SECRET:
        log.error(
            f"[Webhook] REJECTED — WEBHOOK_SECRET not set. "
            f"Caller {request.remote_addr}. Set WEBHOOK_SECRET in .env to enable webhook."
        )
        return jsonify({
            "status": "rejected",
            "reason": "webhook auth not configured (WEBHOOK_SECRET unset)",
        }), 503  # Service Unavailable — config error, not client error
    if secret != WEBHOOK_SECRET:
        log.warning(f"[Webhook] Rejected — invalid secret from {request.remote_addr}")
        return jsonify({"status": "rejected", "reason": "invalid secret"}), 403

    payload = request.get_json(silent=True)
    if not payload:
        # TradingView কখনো plain text/non-strict JSON পাঠাতে পারে
        raw = request.data.decode("utf-8", errors="ignore")
        log.error(f"[Webhook] JSON parse failed. Raw body: {raw[:300]}")
        return jsonify({"status": "error", "reason": "invalid json"}), 400

    log.info(f"[Webhook] Received: {payload}")

    try:
        from server.signal_pipeline import SignalPipeline
        pipeline = SignalPipeline.get_instance()
        result = pipeline.process(payload)
        return jsonify({"status": "ok", "result": result}), 200

    except Exception as e:
        log.error(f"[Webhook] Pipeline error: {e}", exc_info=True)
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/webhook/health", methods=["GET"])
def health_check():
    """Uptime monitor (UptimeRobot ইত্যাদি) দিয়ে ping করার জন্য — server বেঁচে আছে কিনা।"""
    return jsonify({"status": "alive"}), 200


if __name__ == "__main__":
    port = int(os.getenv("WEBHOOK_PORT", "5000"))
    log.info(f"[Webhook] Starting server on port {port}")
    # debug=False — production-এ gunicorn/waitress ব্যবহার করো, app.run() না
    app.run(host="0.0.0.0", port=port, debug=False)