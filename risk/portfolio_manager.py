"""
Portfolio Management System
Handles portfolio-level risk, position sizing, and rebalancing
"""
import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import Config

class PortfolioManager:
    """Manages portfolio-level risk and position sizing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        
        # Portfolio state
        self.positions = {}
        self.portfolio_value = self.config.INITIAL_CAPITAL
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Risk parameters
        self.max_portfolio_risk = 0.02  # 2% portfolio risk per day
        self.max_position_risk = 0.01   # 1% risk per position
        self.max_correlation = 0.7      # Max correlation between positions
        
        # Performance tracking
        self.returns_history = []
        self.high_water_mark = self.portfolio_value
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, account_risk: Optional[float] = None) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Use default account risk if not specified
            if account_risk is None:
                account_risk = self.max_position_risk
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                self.logger.warning(f"Risk per unit is zero for {symbol}")
                return 0.0
            
            # Calculate total risk amount
            risk_amount = self.portfolio_value * account_risk
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Apply additional filters
            position_size = self._apply_volatility_filter(symbol, position_size)
            position_size = self._apply_correlation_filter(symbol, position_size)
            
            # Round to appropriate precision
            position_size = round(position_size, 4)
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size} units")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def _apply_volatility_filter(self, symbol: str, position_size: float) -> float:
        """Adjust position size based on volatility"""
        try:
            # Get symbol volatility (would come from data service)
            volatility = self._get_symbol_volatility(symbol)
            
            if volatility == 0:
                return position_size
            
            # Target volatility (e.g., 1% daily)
            target_volatility = 0.01
            
            # Adjust position size
            adjustment_factor = target_volatility / volatility
            adjusted_size = position_size * adjustment_factor
            
            # Limit adjustment to reasonable range
            adjusted_size = max(position_size * 0.5, min(position_size * 2.0, adjusted_size))
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error applying volatility filter: {e}")
            return position_size
    
    def _apply_correlation_filter(self, symbol: str, position_size: float) -> float:
        """Adjust position size based on correlation with existing positions"""
        try:
            if not self.positions:
                return position_size
            
            # Calculate average correlation with existing positions
            correlations = []
            
            for pos_symbol in self.positions.keys():
                if pos_symbol != symbol:
                    corr = self._get_correlation(symbol, pos_symbol)
                    correlations.append(abs(corr))
            
            if not correlations:
                return position_size
            
            avg_correlation = np.mean(correlations)
            
            # Reduce position size if correlation is high
            if avg_correlation > self.max_correlation:
                reduction_factor = 1.0 - ((avg_correlation - self.max_correlation) * 2)
                reduction_factor = max(0.5, reduction_factor)  # Don't reduce by more than 50%
                
                adjusted_size = position_size * reduction_factor
                self.logger.info(f"Reduced position size for {symbol} due to high correlation: {avg_correlation:.2f}")
                
                return adjusted_size
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error applying correlation filter: {e}")
            return position_size
    
    def update_portfolio(self, positions: Dict, market_prices: Dict):
        """Update portfolio value and track performance"""
        try:
            # Calculate current portfolio value
            total_value = 0.0
            
            for symbol, position in positions.items():
                if symbol in market_prices:
                    current_price = market_prices[symbol]
                    position_value = position['size'] * current_price
                    total_value += position_value
                    
                    # Update position with current price
                    position['current_price'] = current_price
                    position['current_value'] = position_value
            
            # Add cash position
            if 'CASH' in positions:
                total_value += positions['CASH']['size']
            
            # Update portfolio value
            old_value = self.portfolio_value
            self.portfolio_value = total_value
            
            # Calculate return
            if old_value > 0:
                daily_return = (self.portfolio_value - old_value) / old_value
                self.returns_history.append({
                    'date': datetime.now(),
                    'return': daily_return,
                    'portfolio_value': self.portfolio_value
                })
            
            # Update high-water mark
            if self.portfolio_value > self.high_water_mark:
                self.high_water_mark = self.portfolio_value
            
            # Update positions
            self.positions = positions
            
            self.logger.info(f"Portfolio updated - Value: ${self.portfolio_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def calculate_portfolio_risk(self) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not self.positions or len(self.returns_history) < 2:
                return {'status': 'insufficient_data'}
            
            # Convert returns history to DataFrame
            returns_df = pd.DataFrame(self.returns_history)
            returns_series = returns_df['return']
            
            # Calculate risk metrics
            metrics = {
                'portfolio_value': self.portfolio_value,
                'daily_volatility': returns_series.std(),
                'annualized_volatility': returns_series.std() * math.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns_series),
                'sortino_ratio': self._calculate_sortino_ratio(returns_series),
                'max_drawdown': self._calculate_max_drawdown(),
                'value_at_risk': self._calculate_var(returns_series, 0.95),
                'conditional_var': self._calculate_cvar(returns_series, 0.95),
                'position_count': len([p for p in self.positions.values() if p['size'] != 0]),
                'timestamp': datetime.now().isoformat()
            }
            
            # Position-level risk
            position_risks = {}
            
            for symbol, position in self.positions.items():
                if position['size'] != 0:
                    position_risk = self._calculate_position_risk(symbol, position)
                    position_risks[symbol] = position_risk
            
            metrics['position_risks'] = position_risks
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return {'error': str(e)}
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - self.risk_free_rate / 252
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
            
            return sortino
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.returns_history:
                return 0.0
            
            # Convert to cumulative returns
            returns_series = pd.Series([r['return'] for r in self.returns_history])
            cumulative = (1 + returns_series).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            return drawdown.min()
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate VaR using historical method
            var = np.percentile(returns, (1 - confidence) * 100)
            
            return var
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Calculate VaR
            var = self._calculate_var(returns, confidence)
            
            # Calculate CVaR as mean of returns below VaR
            cvar = returns[returns <= var].mean()
            
            return cvar
            
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def _calculate_position_risk(self, symbol: str, position: Dict) -> Dict:
        """Calculate risk metrics for individual position"""
        try:
            current_value = position['current_value']
            entry_price = position['entry_price']
            current_price = position['current_price']
            
            # Calculate unrealized P&L
            unrealized_pnl = (current_price - entry_price) * position['size']
            
            # Calculate position weight
            weight = current_value / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Get symbol volatility
            volatility = self._get_symbol_volatility(symbol)
            
            # Calculate position VaR
            position_var = current_value * volatility * 1.65  # 95% confidence
            
            return {
                'symbol': symbol,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'weight': weight,
                'volatility': volatility,
                'var_95': position_var,
                'entry_price': entry_price,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            return {}
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol (would integrate with data service)"""
        # This would normally fetch from a data service
        # For now, return a default value based on symbol type
        if 'JPY' in symbol:
            return 0.008  # JPY pairs typically lower volatility
        elif 'USD' in symbol:
            return 0.006
        else:
            return 0.010  # Default volatility
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        # This would normally calculate from historical data
        # For now, return a default correlation based on symbol relationships
        if symbol1 == symbol2:
            return 1.0
        
        # USD pairs tend to be negatively correlated
        if 'USD' in symbol1 and 'USD' in symbol2:
            return -0.3
        
        # Same currency pairs are highly correlated
        if symbol1.split('/')[0] == symbol2.split('/')[0]:
            return 0.7
        
        # Default correlation
        return 0.2
    
    def rebalance_portfolio(self, target_weights: Dict) -> Dict:
        """Rebalance portfolio to target weights"""
        try:
            self.logger.info("Starting portfolio rebalancing")
            
            # Calculate current weights
            current_weights = {}
            
            for symbol, position in self.positions.items():
                if position['current_value'] > 0:
                    weight = position['current_value'] / self.portfolio_value
                    current_weights[symbol] = weight
            
            # Calculate required trades
            trades = {}
            
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight
                
                # Calculate trade value
                trade_value = weight_diff * self.portfolio_value
                
                # Get current price
                current_price = self.positions.get(symbol, {}).get('current_price', 0)
                
                if current_price > 0:
                    # Calculate trade size
                    trade_size = trade_value / current_price
                    
                    # Round to appropriate precision
                    trade_size = round(trade_size, 4)
                    
                    trades[symbol] = {
                        'action': 'buy' if trade_size > 0 else 'sell',
                        'size': abs(trade_size),
                        'value': abs(trade_value),
                        'current_weight': current_weight,
                        'target_weight': target_weight
                    }
            
            # Execute trades (would normally go through broker)
            self.logger.info(f"Rebalancing trades: {trades}")
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return {}
    
    def check_risk_limits(self) -> Dict:
        """Check if portfolio is within risk limits"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            
            if 'error' in risk_metrics:
                return {'status': 'error', 'message': risk_metrics['error']}
            
            # Check various risk limits
            checks = {
                'portfolio_var': risk_metrics.get('value_at_risk', 0) > self.max_portfolio_risk,
                'position_count': risk_metrics.get('position_count', 0) > self.config.MAX_POSITIONS,
                'max_drawdown': risk_metrics.get('max_drawdown', 0) < -0.15,  # 15% max drawdown
                'concentration': self._check_concentration()
            }
            
            # Determine overall status
            if any(checks.values()):
                return {
                    'status': 'limit_exceeded',
                    'checks': checks,
                    'actions_required': self._generate_risk_reduction_actions(checks)
                }
            else:
                return {
                    'status': 'within_limits',
                    'checks': checks
                }
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_concentration(self) -> bool:
        """Check for position concentration"""
        try:
            if not self.positions:
                return False
            
            # Calculate position weights
            weights = []
            
            for position in self.positions.values():
                if position['current_value'] > 0:
                    weight = position['current_value'] / self.portfolio_value
                    weights.append(weight)
            
            if not weights:
                return False
            
            # Check if any position exceeds limit
            max_weight = max(weights)
            concentration_limit = 0.25  # 25% max per position
            
            return max_weight > concentration_limit
            
        except Exception as e:
            self.logger.error(f"Error checking concentration: {e}")
            return False
    
    def _generate_risk_reduction_actions(self, exceeded_checks: Dict) -> List[str]:
        """Generate actions to reduce portfolio risk"""
        actions = []
        
        if exceeded_checks.get('portfolio_var'):
            actions.append("Reduce overall portfolio risk by closing or reducing positions")
        
        if exceeded_checks.get('position_count'):
            actions.append("Close some positions to reduce position count")
        
        if exceeded_checks.get('max_drawdown'):
            actions.append("Reduce position sizes to limit further drawdown")
        
        if exceeded_checks.get('concentration'):
            actions.append("Reduce concentrated positions to improve diversification")
        
        return actions

# Singleton instance
portfolio_manager = PortfolioManager()