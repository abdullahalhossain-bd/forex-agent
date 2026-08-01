import os

print("=== FOREX AI DIAGNOSTIC ===")

print("\nENV:")
for k,v in os.environ.items():
    if "MODE" in k or "TRADE" in k or "RISK" in k:
        print(k, "=", v)

print("\nChecking modules...")

modules=[
"agents.decision_agent",
"risk.risk_engine",
"execution",
"signals"
]

for m in modules:
    try:
        __import__(m)
        print("OK:",m)
    except Exception as e:
        print("FAIL:",m,e)

print("\nDiagnostic complete")