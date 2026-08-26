import sys
sys.path.insert(0, r'd:\Projects\forex')
from core.confidence_breakdown import build_confidence_breakdown, reconcile_confidence_breakdown

def show_case(name, rule_conf, llm_conf, pre, post):
    cb = build_confidence_breakdown(
        direction='BUY',
        rule_confidence=rule_conf,
        llm_signal='BUY',
        llm_confidence=llm_conf,
        sentiment_boost=0.0,
        ind_ctx={'trend':'STRONG_UP','rsi_signal':'BULL','macd_cross':'UP'},
        sr_ctx={'dist_to_resistance_pips':8.6,'dist_to_support_pips':0},
        liquidity_ctx={'allowed': True}
    )
    d = cb.to_dict()
    perm = {
        'confidence_pre_penalty': pre,
        'confidence_post_penalty': post,
        'entry_quality_detail': {
            'results': [
                {'flag_name': 'rejection_psychology', 'passed': False, 'confidence_penalty': round(pre-post)}
            ]
        }
    }
    rc = reconcile_confidence_breakdown(d, perm)
    print('---', name, '---')
    print('headline:', post)
    print('lines:')
    for l in rc.to_telegram_lines():
        print(' ', l)
    print()

# GBPCAD: pre=63.0 post=62.0 (from logs)
show_case('GBPCAD', 81.0, 81.0, 63.0, 62.0)
# USDCAD: pre=47.2 post=47.2 (from logs)
show_case('USDCAD', 76.0, 76.0, 47.2, 47.2)
