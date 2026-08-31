from agents.decision_agent import DecisionAgent
agent = DecisionAgent()
analysis_out = {
    'final_signal': 'WAIT',
    'signal': {'signal': 'WAIT', 'confidence': 0},
    'llm': {'signal': 'WAIT', 'confidence': 0, '_llm_unavailable': True},
    'master_ctx': {'master_signal': 'BUY', 'master_confidence': 90, '_llm_unavailable': True},
    'sentiment_ctx': {'sentiment_bias': 'NEUTRAL', 'sentiment_score': 0},
    'conflict': {'has_conflict': False, 'confidence_adjustment': 0},
    'ensemble': {},
    'rl_agent': {},
    'unified_signal': {},
    'session_ctx': {},
    'confluence': {},
    'news': {'trade_allowed': True},
}
market_out = {'symbol': 'EURUSD', 'timeframe': 'M15', 'regime': {'regime': 'NORMAL'}, 'ind_ctx': {'close': 1.0850}}
risk_out = {'approved': True, 'entry': 1.0850, 'sl_price': 1.0830, 'tp_price': 1.0890, 'lot': 0.05, 'rr_ratio': 1.5, 'is_placeholder': False}
result = agent.decide(market_out, analysis_out, risk_out)
print(result)
print('decision', result['decision'])
