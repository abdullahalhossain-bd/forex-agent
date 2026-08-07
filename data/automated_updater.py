"""
Automated Forex Data Update System
Handles daily data refresh, validation, and gap filling
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import yfinance as yf

# OANDA is optional — only needed if OANDA_API_KEY is configured
try:
    import oandapyV20
    from oandapyV20.endpoints import instruments, accounts
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

from config import Config

class ForexDataUpdater:
    """Automated daily data update system with validation and error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.data_dir = os.path.join(self.config.DATA_DIR, 'forex')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize OANDA API client if credentials available
        self.oanda_client = None
        if OANDA_AVAILABLE and hasattr(self.config, 'OANDA_API_KEY') and self.config.OANDA_API_KEY:
            try:
                self.oanda_client = oandapyV20.API(
                    access_token=self.config.OANDA_API_KEY,
                    environment="practice"
                )
                self.logger.info("OANDA API client initialized successfully")
            except Exception as e:
                self.logger.error(f"OANDA API initialization failed: {e}")
    
    def update_all_pairs(self, pairs: List[str] = None) -> Dict[str, bool]:
        """Update all configured currency pairs with latest data"""
        if pairs is None:
            pairs = self.config.FOREX_PAIRS
        
        results = {}
        for pair in pairs:
            try:
                success = self.update_single_pair(pair)
                results[pair] = success
                if success:
                    self.logger.info(f"Successfully updated {pair}")
                else:
                    self.logger.warning(f"Failed to update {pair}")
            except Exception as e:
                self.logger.error(f"Error updating {pair}: {e}")
                results[pair] = False
        
        return results
    
    def update_single_pair(self, pair: str, days_back: int = 30) -> bool:
        """Update a single currency pair with validation and gap filling"""
        try:
            # Get latest existing date
            existing_data = self.load_existing_data(pair)
            latest_date = existing_data.index.max() if not existing_data.empty else None
            
            # Determine update window
            end_date = datetime.now(timezone.utc)
            if latest_date:
                start_date = latest_date - timedelta(days=days_back)
            else:
                start_date = end_date - timedelta(days=365*2)  # 2 years history
            
            # Fetch new data
            new_data = self.fetch_forex_data(pair, start_date, end_date)
            
            if new_data.empty:
                self.logger.warning(f"No new data fetched for {pair}")
                return False
            
            # Validate and clean data
            validated_data = self.validate_and_clean_data(new_data, pair)
            
            # Merge with existing data
            if not existing_data.empty:
                combined_data = pd.concat([existing_data, validated_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
            else:
                combined_data = validated_data
            
            # Save updated data
            self.save_data(pair, combined_data)
            
            # Fill any gaps
            self.fill_data_gaps(pair, combined_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating {pair}: {e}")
            return False
    
    def fetch_forex_data(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch forex data from multiple sources with fallback"""
        # Try OANDA first if available
        if self.oanda_client:
            try:
                data = self._fetch_from_oanda(pair, start_date, end_date)
                if not data.empty:
                    self.logger.info(f"Data fetched from OANDA for {pair}")
                    return data
            except Exception as e:
                self.logger.warning(f"OANDA fetch failed for {pair}: {e}")
        
        # Fallback to Yahoo Finance
        try:
            data = self._fetch_from_yahoo(pair, start_date, end_date)
            if not data.empty:
                self.logger.info(f"Data fetched from Yahoo Finance for {pair}")
                return data
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed for {pair}: {e}")
        
        # Final fallback: generate synthetic data for testing
        self.logger.warning(f"Using synthetic data for {pair}")
        return self._generate_synthetic_data(pair, start_date, end_date)
    
    def _fetch_from_oanda(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from OANDA API"""
        try:
            # Convert pair format (EUR/USD -> EUR_USD)
            oanda_pair = pair.replace('/', '_')
            
            params = {
                "from": start_date.strftime('%Y-%m-%d'),
                "to": end_date.strftime('%Y-%m-%d'),
                "granularity": "D"  # Daily data
            }
            
            r = instruments.InstrumentsCandles(instrument=oanda_pair, params=params)
            self.oanda_client.request(r)
            
            # Parse response
            candles = r.response.get('candles', [])
            data = []
            for candle in candles:
                data.append({
                    'Date': pd.to_datetime(candle['time'], unit='s'),
                    'Open': float(candle['mid']['o']),
                    'High': float(candle['mid']['h']),
                    'Low': float(candle['mid']['l']),
                    'Close': float(candle['mid']['c']),
                    'Volume': int(candle['volume'])
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"OANDA API error: {e}")
            raise
    
    def _fetch_from_yahoo(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            # Skip commodity pairs (XAUUSD, XAGUSD) — not available on Yahoo Finance in this format
            if pair.upper() in ("XAUUSD", "XAGUSD"):
                return pd.DataFrame()
            
            # Yahoo uses different format (EURUSD=X)
            yahoo_pair = pair.replace('/', '=X')
            
            data = yf.download(yahoo_pair, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Rename columns to standard format
                data = data.rename(columns={
                    'Open': 'Open',
                    'High': 'High', 
                    'Low': 'Low',
                    'Close': 'Close',
                    'Adj Close': 'Adj_Close',
                    'Volume': 'Volume'
                })
                
                # Keep only relevant columns
                columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                data = data[[col for col in columns if col in data.columns]]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error: {e}")
            raise
    
    def _generate_synthetic_data(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Return empty DataFrame — synthetic data must never enter the real pipeline."""
        self.logger.warning(
            f"All real data sources failed for {pair}. "
            f"Refusing to generate synthetic data to avoid contamination. Returning empty DataFrame."
        )
        return pd.DataFrame()
    
    def validate_and_clean_data(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Validate and clean forex data"""
        if data.empty:
            return data
        
        # Make a copy to avoid SettingWithCopyWarning
        data = data.copy()
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns for {pair}: {missing_cols}")
            # Try to infer missing columns
            if 'Adj_Close' in data.columns and 'Close' not in data.columns:
                data['Close'] = data['Adj_Close']
                self.logger.info("Inferred 'Close' from 'Adj_Close'")
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='last')]
        
        # Handle missing values
        if data.isnull().any().any():
            self.logger.warning(f"Missing values detected in {pair}")
            # Forward fill then backward fill
            data.ffill(inplace=True)
            data.bfill(inplace=True)
        
        # Validate price relationships
        invalid_rows = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])
        )
        
        if invalid_rows.any():
            self.logger.warning(f"Found {invalid_rows.sum()} invalid price rows in {pair}")
            # Fix invalid rows by adjusting prices
            data.loc[invalid_rows, 'High'] = data.loc[invalid_rows, ['Open', 'Close']].max(axis=1)
            data.loc[invalid_rows, 'Low'] = data.loc[invalid_rows, ['Open', 'Close']].min(axis=1)
        
        # Ensure chronological order
        data.sort_index(inplace=True)
        
        # Add derived features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data
    
    def fill_data_gaps(self, pair: str, data: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in time series data"""
        if data.empty:
            return data
        
        # Check for missing dates
        all_dates = pd.bdate_range(start=data.index.min(), end=data.index.max())
        missing_dates = all_dates.difference(data.index)
        
        if len(missing_dates) > 0:
            self.logger.info(f"Found {len(missing_dates)} missing dates for {pair}")
            
            # Create DataFrame with missing dates
            missing_df = pd.DataFrame(index=missing_dates, columns=data.columns)
            
            # Forward fill from last known values
            combined = pd.concat([data, missing_df])
            combined.sort_index(inplace=True)
            combined.ffill(inplace=True)
            
            # Also fill any remaining NaNs with backward fill
            combined.bfill(inplace=True)
            
            # Save updated data
            self.save_data(pair, combined)
            
            return combined
        
        return data
    
    def load_existing_data(self, pair: str) -> pd.DataFrame:
        """Load existing data for a currency pair"""
        filepath = os.path.join(self.data_dir, f"{pair.replace('/', '_')}_daily.csv")
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                self.logger.info(f"Loaded existing data for {pair}: {len(data)} records")
                return data
            except Exception as e:
                self.logger.error(f"Error loading data for {pair}: {e}")
                return pd.DataFrame()
        else:
            self.logger.info(f"No existing data found for {pair}")
            return pd.DataFrame()
    
    def save_data(self, pair: str, data: pd.DataFrame) -> None:
        """Save data for a currency pair"""
        filepath = os.path.join(self.data_dir, f"{pair.replace('/', '_')}_daily.csv")
        
        try:
            data.to_csv(filepath)
            self.logger.info(f"Saved {len(data)} records for {pair} to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data for {pair}: {e}")
            raise
    
    def get_data_status(self) -> Dict[str, Dict]:
        """Get status of all currency pair data"""
        status = {}
        pairs = self.config.FOREX_PAIRS
        
        for pair in pairs:
            filepath = os.path.join(self.data_dir, f"{pair.replace('/', '_')}_daily.csv")
            
            if os.path.exists(filepath):
                try:
                    data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                    status[pair] = {
                        'last_date': data.index.max().strftime('%Y-%m-%d'),
                        'record_count': len(data),
                        'file_size': os.path.getsize(filepath),
                        'status': 'OK'
                    }
                except Exception as e:
                    status[pair] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
            else:
                status[pair] = {
                    'status': 'MISSING',
                    'error': 'Data file not found'
                }
        
        return status

# Singleton instance
data_updater = ForexDataUpdater()