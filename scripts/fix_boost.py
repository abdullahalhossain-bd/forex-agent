"""Fix consecutive loss boost bug: min() was reducing scores instead of only capping boosts."""

PATH = "/home/z/my-project/forex-agent/core/loss_rejection_engine/layer1_structural_filters.py"

with open(PATH, 'r') as f:
    content = f.read()

old_boost = """        # Consecutive loss boost: capped below hard-block threshold
        # Only pushes sub-threshold scores into WARN zone
        if cl >= 5: s = min(65, s + 20)
        elif cl >= 4: s = min(65, s + 15)
        elif cl >= 3: s = min(55, s + 10)"""

new_boost = """        # Consecutive loss boost: only applies to sub-threshold scores
        # Never reduces a score that's already at hard-block level (>=70)
        if s < 70:
            if cl >= 5: s = min(69, s + 20)
            elif cl >= 4: s = min(69, s + 15)
            elif cl >= 3: s = min(69, s + 10)"""

assert old_boost in content, "old boost not found!"
content = content.replace(old_boost, new_boost, 1)

with open(PATH, 'w') as f:
    f.write(content)

print("Boost bug fixed: no longer reduces hard-block scores")