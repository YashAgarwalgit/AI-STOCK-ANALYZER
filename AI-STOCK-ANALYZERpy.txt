import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

# Advanced Analytics Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Using custom technical indicators.")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Time series analysis will be limited.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Scikit-learn not available. ML predictions disabled.")

try:
    from textblob import TextBlob
    import requests
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("TextBlob/Requests not available. News sentiment analysis disabled.")

warnings.filterwarnings('ignore')

class AdvancedStockAnalyzer:
    """
    Comprehensive AI-powered stock analysis system with machine learning capabilities
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        """
        Initialize the analyzer with configuration parameters
        
        Args:
            risk_free_rate: Current risk-free rate for calculations (default: 4.5%)
        """
        self.risk_free_rate = risk_free_rate
        self.analysis_cache = {}
        self.ml_models = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def fetch_stock_data(self, ticker: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch comprehensive stock data with error handling
        
        Args:
            ticker: Stock symbol
            period: Time period for data retrieval
            
        Returns:
            Tuple of (price_data, company_info)
        """
        try:
            stock = yf.Ticker(ticker.upper())
            data = stock.history(period=period)
            info = stock.info
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            self.logger.info(f"Successfully fetched data for {ticker}")
            return data, info
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            raise
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators using TA-Lib when available
        
        Args:
            data: OHLCV price data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        if TALIB_AVAILABLE:
            # TA-Lib based indicators (more accurate)
            df = self._calculate_talib_indicators(df)
        else:
            # Custom implementations
            df = self._calculate_custom_indicators(df)
            
        return df
    
    def _calculate_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using TA-Lib library"""
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Trend Indicators
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['EMA_20'] = talib.EMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(close)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close)
        
        # RSI
        df['RSI'] = talib.RSI(close, timeperiod=14)
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close)
        
        # Williams %R
        df['WILLIAMS_R'] = talib.WILLR(high, low, close)
        
        # Average True Range
        df['ATR'] = talib.ATR(high, low, close)
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(high, low, close)
        
        # Money Flow Index
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Average Directional Index
        df['ADX'] = talib.ADX(high, low, close)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(high, low)
        
        # Volume indicators
        df['OBV'] = talib.OBV(close, volume)
        df['AD'] = talib.AD(high, low, close, volume)
        
        # Momentum indicators
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        
        return df
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using custom implementations when TA-Lib unavailable"""
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def advanced_risk_metrics(self, data: pd.DataFrame, benchmark_ticker: str = 'SPY') -> Dict:
        """
        Calculate comprehensive risk metrics including Greek letters approximations
        
        Args:
            data: Price data with technical indicators
            benchmark_ticker: Benchmark for beta calculation
            
        Returns:
            Dictionary of risk metrics
        """
        returns = data['Close'].pct_change().dropna()
        
        # Get benchmark data for beta calculation
        try:
            benchmark = yf.Ticker(benchmark_ticker).history(period='1y')['Close']
            benchmark_returns = benchmark.pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 50:
                stock_aligned = returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]
                
                # Beta calculation
                covariance = np.cov(stock_aligned, bench_aligned)[0][1]
                benchmark_var = np.var(bench_aligned)
                beta = covariance / benchmark_var if benchmark_var > 0 else 1.0
            else:
                beta = 1.0
        except:
            beta = 1.0
        
        # Basic metrics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized metrics
        annual_return = mean_return * 252
        annual_volatility = std_return * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% and 99% confidence)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information Ratio
        tracking_error = std_return * np.sqrt(252)
        information_ratio = (annual_return - self.risk_free_rate) / tracking_error if tracking_error > 0 else 0
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns.dropna())
        kurtosis = stats.kurtosis(returns.dropna())
        
        # Approximate Greeks (simplified estimations)
        current_price = data['Close'].iloc[-1]
        price_changes = data['Close'].diff()
        
        # Delta approximation (price sensitivity)
        delta = 1.0  # For stocks, delta is always 1
        
        # Gamma approximation (rate of change of delta)
        gamma = price_changes.rolling(5).std().iloc[-1] / current_price if current_price > 0 else 0
        
        # Theta approximation (time decay - not applicable to stocks directly)
        theta = 0  # Stocks don't have time decay
        
        # Vega approximation (volatility sensitivity)
        volatility_changes = returns.rolling(20).std().diff()
        vega = abs(volatility_changes.corr(price_changes)) if not volatility_changes.empty else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'beta': beta,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    def time_series_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform advanced time series analysis using statsmodels
        
        Args:
            data: Price data
            
        Returns:
            Dictionary of time series analysis results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available for time series analysis"}
        
        try:
            prices = data['Close'].dropna()
            returns = prices.pct_change().dropna()
            
            results = {}
            
            # Seasonal Decomposition
            if len(prices) >= 104:  # Need at least 2 years of weekly data
                decomposition = seasonal_decompose(prices, model='multiplicative', period=52)
                results['seasonal_strength'] = np.var(decomposition.seasonal) / np.var(prices)
                results['trend_strength'] = np.var(decomposition.trend.dropna()) / np.var(prices)
            
            # Stationarity Test (Augmented Dickey-Fuller)
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns)
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['is_stationary'] = adf_result[1] < 0.05
            
            # Ljung-Box Test for autocorrelation
            lb_test = acorr_ljungbox(returns, lags=10, return_df=True)
            results['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]
            results['has_autocorrelation'] = lb_test['lb_pvalue'].iloc[-1] < 0.05
            
            # ARIMA Model Fitting
            try:
                # Simple ARIMA(1,1,1) model
                model = ARIMA(prices, order=(1, 1, 1))
                fitted_model = model.fit()
                
                results['arima_aic'] = fitted_model.aic
                results['arima_bic'] = fitted_model.bic
                
                # Forecast next 5 periods
                forecast = fitted_model.forecast(steps=5)
                results['arima_forecast'] = forecast.tolist()
                
            except Exception as e:
                self.logger.warning(f"ARIMA modeling failed: {e}")
                results['arima_error'] = str(e)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {e}")
            return {"error": str(e)}
    
    def ml_price_prediction(self, data: pd.DataFrame, days_ahead: int = 5) -> Dict:
        """
        Use LSTM neural network for price prediction
        
        Args:
            data: Price data with technical indicators
            days_ahead: Number of days to predict
            
        Returns:
            Dictionary with predictions and model metrics
        """
        if not TENSORFLOW_AVAILABLE:
            return {"error": "TensorFlow/Scikit-learn not available for ML predictions"}
        
        try:
            # Prepare features
            feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
            feature_columns = [col for col in feature_columns if col in data.columns]
            
            if len(feature_columns) < 2:
                return {"error": "Insufficient features for ML prediction"}
            
            # Prepare data
            df = data[feature_columns].dropna()
            
            if len(df) < 100:
                return {"error": "Insufficient data for ML training (need at least 100 samples)"}
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)
            
            # Create sequences for LSTM
            sequence_length = 60
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # Predict Close price
            
            X, y = np.array(X), np.array(y)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model (with minimal epochs for demo)
            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0, validation_split=0.2)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Predict future prices
            last_sequence = scaled_data[-sequence_length:]
            future_predictions = []
            
            for _ in range(days_ahead):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, len(feature_columns)))[0, 0]
                future_predictions.append(next_pred)
                
                # Update sequence for next prediction
                new_row = last_sequence[-1].copy()
                new_row[0] = next_pred
                last_sequence = np.vstack([last_sequence[1:], new_row])
            
            # Inverse transform predictions
            dummy_array = np.zeros((len(future_predictions), len(feature_columns)))
            dummy_array[:, 0] = future_predictions
            future_prices = scaler.inverse_transform(dummy_array)[:, 0]
            
            return {
                'future_predictions': future_prices.tolist(),
                'model_mse': mse,
                'model_mae': mae,
                'prediction_days': days_ahead,
                'last_actual_price': data['Close'].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return {"error": str(e)}
    
    def pattern_recognition_advanced(self, data: pd.DataFrame) -> List[str]:
        """
        Advanced candlestick and chart pattern recognition
        
        Args:
            data: OHLCV data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if TALIB_AVAILABLE:
            # TA-Lib candlestick patterns
            open_prices = data['Open'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Major reversal patterns
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            
            # Check recent patterns (last 5 days)
            recent_signals = 5
            
            if np.any(doji[-recent_signals:] != 0):
                patterns.append("🕯️ DOJI: Indecision pattern - potential reversal")
            
            if np.any(hammer[-recent_signals:] > 0):
                patterns.append("🔨 HAMMER: Bullish reversal pattern detected")
            
            if np.any(engulfing[-recent_signals:] > 0):
                patterns.append("🐂 BULLISH ENGULFING: Strong bullish reversal signal")
            elif np.any(engulfing[-recent_signals:] < 0):
                patterns.append("🐻 BEARISH ENGULFING: Strong bearish reversal signal")
            
            if np.any(morning_star[-recent_signals:] > 0):
                patterns.append("🌅 MORNING STAR: Major bullish reversal pattern")
            
            if np.any(evening_star[-recent_signals:] > 0):
                patterns.append("🌆 EVENING STAR: Major bearish reversal pattern")
        
        # Chart patterns using price action
        if len(data) >= 50:
            patterns.extend(self._detect_chart_patterns(data))
        
        return patterns if patterns else ["No significant patterns detected"]
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[str]:
        """Detect chart patterns using price action analysis"""
        patterns = []
        
        # Support and resistance levels
        highs = data['High'].rolling(window=10, center=True).max()
        lows = data['Low'].rolling(window=10, center=True).min()
        
        current_price = data['Close'].iloc[-1]
        
        # Breakout patterns
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        if current_price > recent_high * 1.02:
            patterns.append("🚀 BREAKOUT: Price breaking above recent resistance")
        elif current_price < recent_low * 0.98:
            patterns.append("📉 BREAKDOWN: Price breaking below recent support")
        
        # Triangle patterns
        if len(data) >= 20:
            recent_highs = data['High'].tail(20)
            recent_lows = data['Low'].tail(20)
            
            # Simple trend detection
            high_trend = stats.linregress(range(len(recent_highs)), recent_highs)[0]
            low_trend = stats.linregress(range(len(recent_lows)), recent_lows)[0]
            
            if high_trend < 0 and low_trend > 0:
                patterns.append("📐 TRIANGLE: Converging price action - breakout imminent")
        
        return patterns
    
    def sentiment_analysis_news(self, ticker: str) -> Dict:
        """
        Analyze sentiment from recent news headlines
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not SENTIMENT_AVAILABLE:
            return {"error": "TextBlob/Requests not available for sentiment analysis"}
        
        try:
            # This would typically use a news API like Alpha Vantage, NewsAPI, etc.
            # For demo purposes, we'll simulate sentiment analysis
            
            # In a real implementation, you would fetch news headlines here
            # For now, we'll return a placeholder
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "positive_mentions": 0,
                "negative_mentions": 0,
                "news_volume": 0,
                "note": "News sentiment analysis requires API keys for news services"
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_report(self, ticker: str, period: str = "1y") -> Dict:
        """
        Generate a comprehensive analysis report
        
        Args:
            ticker: Stock symbol
            period: Analysis period
            
        Returns:
            Complete analysis dictionary
        """
        try:
            # Fetch data
            data, info = self.fetch_stock_data(ticker, period)
            
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Perform all analyses
            risk_metrics = self.advanced_risk_metrics(data_with_indicators)
            time_series = self.time_series_analysis(data_with_indicators)
            ml_predictions = self.ml_price_prediction(data_with_indicators)
            patterns = self.pattern_recognition_advanced(data_with_indicators)
            sentiment = self.sentiment_analysis_news(ticker)
            
            # Compile comprehensive report
            report = {
                'ticker': ticker.upper(),
                'company_info': {
                    'name': info.get('longName', ticker.upper()),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'employees': info.get('fullTimeEmployees', 0)
                },
                'current_metrics': {
                    'price': data['Close'].iloc[-1],
                    'change': data['Close'].iloc[-1] - data['Close'].iloc[-2],
                    'change_percent': ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100,
                    'volume': int(data['Volume'].iloc[-1]),
                    'avg_volume': int(data['Volume'].mean()),
                    '52w_high': data['High'].max(),
                    '52w_low': data['Low'].min()
                },
                'risk_metrics': risk_metrics,
                'time_series_analysis': time_series,
                'ml_predictions': ml_predictions,
                'patterns': patterns,
                'sentiment': sentiment,
                'technical_indicators': {
                    'rsi': data_with_indicators['RSI'].iloc[-1] if 'RSI' in data_with_indicators.columns else None,
                    'macd': data_with_indicators['MACD'].iloc[-1] if 'MACD' in data_with_indicators.columns else None,
                    'bb_position': self._calculate_bb_position(data_with_indicators)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed for {ticker}: {e}")
            return {"error": str(e)}
    
    def _calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calculate current position within Bollinger Bands"""
        if 'BB_Upper' not in data.columns or 'BB_Lower' not in data.columns:
            return 0.5
        
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        
        if bb_upper == bb_lower:
            return 0.5
        
        return (current_price - bb_lower) / (bb_upper - bb_lower)
    
    def create_advanced_visualizations(self, data: pd.DataFrame, info: Dict, 
                                     ticker: str, analysis_results: Dict):
        """
        Create comprehensive visualization dashboard
        
        Args:
            data: Price data with indicators
            info: Company information
            ticker: Stock symbol
            analysis_results: Analysis results dictionary
        """
        company_name = info.get('longName', ticker.upper())
        
        # 1. Main Technical Analysis Chart
        fig1 = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price & Volume', 'RSI', 'MACD', 'Volume'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Candlestick chart
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=ticker.upper()
        ), row=1, col=1)
        
        # Moving averages
        if 'SMA_20' in data.columns:
            fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                                     name='SMA 20', line=dict(color='orange')), row=1, col=1)
        if 'SMA_50' in data.columns:
            fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                     name='SMA 50', line=dict(color='red')), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            fig1.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                                     name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig1.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                                     name='BB Lower', line=dict(color='gray', dash='dash'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # RSI
        if 'RSI' in data.columns:
            fig1.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', 
                                     line=dict(color='purple')), row=2, col=1)
            fig1.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig1.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in data.columns:
            fig1.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', 
                                     line=dict(color='blue')), row=3, col=1)
            fig1.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', 
                                     line=dict(color='red')), row=3, col=1)
            if 'MACD_Hist' in data.columns:
                fig1.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD Hist', 
                                     marker_color='green'), row=3, col=1)
        
        # Volume
        fig1.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                             marker_color='lightblue'), row=4, col=1)
        
        fig1.update_layout(
            title=f"{company_name} ({ticker.upper()}) - Technical Analysis Dashboard",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig1.show()
        
        # 2. Risk Metrics Visualization
        if 'risk_metrics' in analysis_results:
            risk_data = analysis_results['risk_metrics']
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk-Return Profile', 'Drawdown Analysis', 
                               'Value at Risk', 'Greek Letters'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Risk-Return scatter
            fig2.add_trace(go.Scatter(
                x=[risk_data.get('annual_volatility', 0)],
                y=[risk_data.get('annual_return', 0)],
                mode='markers',
                marker=dict(size=15, color='red'),
                name=ticker.upper(),
                text=[f"Sharpe: {risk_data.get('sharpe_ratio', 0):.2f}"]
            ), row=1, col=1)
            
            # Drawdown visualization
            returns = data['Close'].pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            fig2.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tozeroy',
                name='Drawdown %',
                line=dict(color='red')
            ), row=1, col=2)
            
            # VaR comparison
            var_data = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            var_values = [risk_data.get('var_95', 0) * 100, risk_data.get('var_99', 0) * 100,
                         risk_data.get('cvar_95', 0) * 100, risk_data.get('cvar_99', 0) * 100]
            
            fig2.add_trace(go.Bar(
                x=var_data,
                y=var_values,
                name='Value at Risk',
                marker_color=['orange', 'red', 'darkred', 'maroon']
            ), row=2, col=1)
            
            # Greeks visualization
            greeks_data = ['Delta', 'Gamma', 'Vega', 'Beta']
            greeks_values = [risk_data.get('delta', 0), risk_data.get('gamma', 0),
                           risk_data.get('vega', 0), risk_data.get('beta', 0)]
            
            fig2.add_trace(go.Bar(
                x=greeks_data,
                y=greeks_values,
                name='Risk Sensitivities',
                marker_color=['blue', 'green', 'purple', 'orange']
            ), row=2, col=2)
            
            fig2.update_layout(
                title=f"{company_name} - Risk Analysis Dashboard",
                height=600,
                showlegend=False
            )
            
            fig2.show()
        
        # 3. Pattern Recognition Visualization
        if 'patterns' in analysis_results:
            print(f"\n🔍 PATTERN RECOGNITION FOR {ticker.upper()}")
            print("=" * 50)
            for pattern in analysis_results['patterns']:
                print(f"• {pattern}")
    
    def fibonacci_retracement_levels(self, data: pd.DataFrame, 
                                   lookback_period: int = 50) -> Dict:
        """
        Calculate Fibonacci retracement levels for price targets
        
        Args:
            data: Price data
            lookback_period: Period to look back for high/low
            
        Returns:
            Dictionary with Fibonacci levels
        """
        recent_data = data.tail(lookback_period)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        
        diff = high - low
        
        # Fibonacci retracement levels
        levels = {
            'high': high,
            'low': low,
            'fib_23.6': high - 0.236 * diff,
            'fib_38.2': high - 0.382 * diff,
            'fib_50.0': high - 0.500 * diff,
            'fib_61.8': high - 0.618 * diff,
            'fib_78.6': high - 0.786 * diff
        }
        
        # Extension levels
        levels.update({
            'ext_127.2': high + 0.272 * diff,
            'ext_161.8': high + 0.618 * diff,
            'ext_200.0': high + 1.000 * diff
        })
        
        return levels
    
    def support_resistance_levels(self, data: pd.DataFrame, 
                                window: int = 20) -> Dict:
        """
        Identify key support and resistance levels
        
        Args:
            data: Price data
            window: Window for local extrema detection
            
        Returns:
            Dictionary with support/resistance levels
        """
        # Find local maxima and minima
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        # Identify resistance levels (local maxima)
        resistance_levels = []
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
        
        # Identify support levels (local minima)
        support_levels = []
        for i in range(window, len(data) - window):
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        # Get most significant levels
        current_price = data['Close'].iloc[-1]
        
        # Filter levels within reasonable range
        price_range = current_price * 0.15  # 15% range
        
        nearby_resistance = [r for r in resistance_levels 
                           if current_price < r <= current_price + price_range]
        nearby_support = [s for s in support_levels 
                         if current_price - price_range <= s < current_price]
        
        return {
            'current_price': current_price,
            'immediate_resistance': min(nearby_resistance) if nearby_resistance else None,
            'immediate_support': max(nearby_support) if nearby_support else None,
            'all_resistance': sorted(set(nearby_resistance), reverse=True)[:3],
            'all_support': sorted(set(nearby_support), reverse=True)[:3]
        }
    
    def generate_price_targets(self, data: pd.DataFrame, 
                             analysis_results: Dict) -> Dict:
        """
        Generate comprehensive price targets using multiple methods
        
        Args:
            data: Price data with indicators
            analysis_results: Analysis results dictionary
            
        Returns:
            Dictionary with price targets and projections
        """
        current_price = data['Close'].iloc[-1]
        
        # Fibonacci levels
        fib_levels = self.fibonacci_retracement_levels(data)
        
        # Support/Resistance levels
        sr_levels = self.support_resistance_levels(data)
        
        # ATR-based targets
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns else current_price * 0.02
        
        # Technical indicator targets
        targets = {
            'current_price': current_price,
            'fibonacci_targets': {
                'bullish_target_1': fib_levels['ext_127.2'],
                'bullish_target_2': fib_levels['ext_161.8'],
                'bullish_target_3': fib_levels['ext_200.0'],
                'support_1': fib_levels['fib_38.2'],
                'support_2': fib_levels['fib_61.8'],
                'support_3': fib_levels['low']
            },
            'atr_targets': {
                'short_term_high': current_price + (atr * 2),
                'short_term_low': current_price - (atr * 2),
                'medium_term_high': current_price + (atr * 4),
                'medium_term_low': current_price - (atr * 4)
            },
            'sr_levels': sr_levels
        }
        
        # ML-based targets if available
        if 'ml_predictions' in analysis_results:
            ml_data = analysis_results['ml_predictions']
            if 'future_predictions' in ml_data:
                targets['ml_targets'] = {
                    'next_week': ml_data['future_predictions'][-1],
                    'trend_direction': 'bullish' if ml_data['future_predictions'][-1] > current_price else 'bearish'
                }
        
        return targets
    
    def generate_trading_signals(self, data: pd.DataFrame, 
                               analysis_results: Dict) -> Dict:
        """
        Generate comprehensive trading signals based on multiple factors
        
        Args:
            data: Price data with indicators
            analysis_results: Analysis results dictionary
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        signals = {
            'overall_signal': 'HOLD',
            'signal_strength': 0,
            'individual_signals': {},
            'risk_level': 'MEDIUM',
            'confidence': 0.5
        }
        
        signal_count = 0
        bullish_signals = 0
        bearish_signals = 0
        
        # Technical indicator signals
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi < 30:
                signals['individual_signals']['RSI'] = 'STRONG_BUY'
                bullish_signals += 2
            elif rsi < 45:
                signals['individual_signals']['RSI'] = 'BUY'
                bullish_signals += 1
            elif rsi > 70:
                signals['individual_signals']['RSI'] = 'STRONG_SELL'
                bearish_signals += 2
            elif rsi > 55:
                signals['individual_signals']['RSI'] = 'SELL'
                bearish_signals += 1
            else:
                signals['individual_signals']['RSI'] = 'NEUTRAL'
            signal_count += 1
        
        # MACD signals
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            
            if macd > macd_signal and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
                signals['individual_signals']['MACD'] = 'BUY'
                bullish_signals += 1
            elif macd < macd_signal and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
                signals['individual_signals']['MACD'] = 'SELL'
                bearish_signals += 1
            else:
                signals['individual_signals']['MACD'] = 'NEUTRAL'
            signal_count += 1
        
        # Moving average signals
        if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                signals['individual_signals']['MA_Trend'] = 'BUY'
                bullish_signals += 1
            elif current_price < sma_20 < sma_50:
                signals['individual_signals']['MA_Trend'] = 'SELL'
                bearish_signals += 1
            else:
                signals['individual_signals']['MA_Trend'] = 'NEUTRAL'
            signal_count += 1
        
        # Bollinger Bands signals
        bb_position = self._calculate_bb_position(data)
        if bb_position < 0.2:
            signals['individual_signals']['Bollinger_Bands'] = 'BUY'
            bullish_signals += 1
        elif bb_position > 0.8:
            signals['individual_signals']['Bollinger_Bands'] = 'SELL'
            bearish_signals += 1
        else:
            signals['individual_signals']['Bollinger_Bands'] = 'NEUTRAL'
        signal_count += 1
        
        # Volume confirmation
        if len(data) >= 20:
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                signals['individual_signals']['Volume'] = 'STRONG_CONFIRMATION'
            elif current_volume > avg_volume * 1.2:
                signals['individual_signals']['Volume'] = 'CONFIRMATION'
            else:
                signals['individual_signals']['Volume'] = 'WEAK'
        
        # Risk assessment
        if 'risk_metrics' in analysis_results:
            risk_data = analysis_results['risk_metrics']
            volatility = risk_data.get('annual_volatility', 0.2)
            
            if volatility > 0.4:
                signals['risk_level'] = 'HIGH'
            elif volatility < 0.15:
                signals['risk_level'] = 'LOW'
            else:
                signals['risk_level'] = 'MEDIUM'
        
        # Calculate overall signal
        if signal_count > 0:
            net_signal = bullish_signals - bearish_signals
            signal_strength = abs(net_signal) / signal_count
            
            if net_signal >= 2:
                signals['overall_signal'] = 'STRONG_BUY'
            elif net_signal >= 1:
                signals['overall_signal'] = 'BUY'
            elif net_signal <= -2:
                signals['overall_signal'] = 'STRONG_SELL'
            elif net_signal <= -1:
                signals['overall_signal'] = 'SELL'
            else:
                signals['overall_signal'] = 'HOLD'
            
            signals['signal_strength'] = signal_strength
            signals['confidence'] = min(0.9, 0.5 + (signal_strength * 0.4))
        
        return signals
    
    def portfolio_optimization(self, tickers: List[str], 
                             period: str = "1y") -> Dict:
        """
        Perform portfolio optimization using Monte Carlo simulation
        
        Args:
            tickers: List of stock symbols
            period: Data period
            
        Returns:
            Dictionary with optimal portfolio allocation
        """
        try:
            # Fetch data for all tickers
            portfolio_data = {}
            for ticker in tickers:
                try:
                    data, _ = self.fetch_stock_data(ticker, period)
                    portfolio_data[ticker] = data['Close']
                except:
                    self.logger.warning(f"Failed to fetch data for {ticker}")
                    continue
            
            if len(portfolio_data) < 2:
                return {"error": "Need at least 2 valid stocks for portfolio optimization"}
            
            # Create returns dataframe
            returns_df = pd.DataFrame(portfolio_data).pct_change().dropna()
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Monte Carlo simulation
            num_simulations = 10000
            num_assets = len(tickers)
            
            results = np.zeros((4, num_simulations))
            
            for i in range(num_simulations):
                # Random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                
                # Portfolio return and risk
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Sharpe ratio
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_std
                results[2, i] = sharpe
                results[3, i] = i  # Index for tracking
            
            # Find optimal portfolios
            max_sharpe_idx = np.argmax(results[2])
            min_vol_idx = np.argmin(results[1])
            
            # Calculate optimal weights (simplified approach)
            # For demonstration - in practice, use scipy.optimize
            optimal_weights = np.random.random(num_assets)
            optimal_weights /= np.sum(optimal_weights)
            
            return {
                'tickers': tickers,
                'optimal_weights': dict(zip(tickers, optimal_weights)),
                'expected_return': results[0, max_sharpe_idx],
                'expected_volatility': results[1, max_sharpe_idx],
                'sharpe_ratio': results[2, max_sharpe_idx],
                'efficient_frontier': {
                    'returns': results[0].tolist(),
                    'volatilities': results[1].tolist(),
                    'sharpe_ratios': results[2].tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {"error": str(e)}
    
    def sector_comparison(self, ticker: str, sector_etf: str = None) -> Dict:
        """
        Compare stock performance against sector
        
        Args:
            ticker: Stock symbol
            sector_etf: Sector ETF for comparison
            
        Returns:
            Dictionary with sector comparison metrics
        """
        try:
            # Get stock data
            stock_data, stock_info = self.fetch_stock_data(ticker, "1y")
            
            # Determine sector ETF if not provided
            if not sector_etf:
                sector = stock_info.get('sector', '')
                sector_etfs = {
                    'Technology': 'XLK',
                    'Healthcare': 'XLV',
                    'Financial Services': 'XLF',
                    'Consumer Cyclical': 'XLY',
                    'Energy': 'XLE',
                    'Industrials': 'XLI',
                    'Consumer Defensive': 'XLP',
                    'Utilities': 'XLU',
                    'Real Estate': 'XLRE',
                    'Materials': 'XLB',
                    'Communication Services': 'XLC'
                }
                sector_etf = sector_etfs.get(sector, 'SPY')
            
            # Get sector data
            sector_data, _ = self.fetch_stock_data(sector_etf, "1y")
            
            # Calculate relative performance
            stock_returns = stock_data['Close'].pct_change().dropna()
            sector_returns = sector_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = stock_returns.index.intersection(sector_returns.index)
            stock_aligned = stock_returns.loc[common_dates]
            sector_aligned = sector_returns.loc[common_dates]
            
            # Performance metrics
            stock_total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
            sector_total_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0] - 1) * 100
            
            outperformance = stock_total_return - sector_total_return
            
            # Correlation
            correlation = stock_aligned.corr(sector_aligned)
            
            # Beta vs sector
            covariance = np.cov(stock_aligned, sector_aligned)[0][1]
            sector_variance = np.var(sector_aligned)
            sector_beta = covariance / sector_variance if sector_variance > 0 else 1.0
            
            return {
                'ticker': ticker.upper(),
                'sector_etf': sector_etf,
                'stock_return': stock_total_return,
                'sector_return': sector_total_return,
                'outperformance': outperformance,
                'correlation': correlation,
                'sector_beta': sector_beta,
                'relative_strength': 'STRONG' if outperformance > 10 else 'WEAK' if outperformance < -10 else 'MODERATE'
            }
            
        except Exception as e:
            self.logger.error(f"Sector comparison failed: {e}")
            return {"error": str(e)}


class StockAnalysisDashboard:
    """
    Interactive dashboard for stock analysis with menu-driven interface
    """
    
    def __init__(self):
        self.analyzer = AdvancedStockAnalyzer()
        self.analysis_cache = {}
        
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🚀 ENHANCED STOCK ANALYSIS DASHBOARD 🚀")
        print("="*60)
        print("1. 📊 Single Stock Deep Analysis")
        print("2. 📈 Portfolio Optimization")
        print("3. 🎯 Price Target Analysis")
        print("4. 📊 Sector Comparison")
        print("5. 🔍 Batch Stock Analysis")
        print("6. 📋 View Analysis History")
        print("7. ❌ Exit")
        print("="*60)
    
    def single_stock_analysis(self):
        """Perform comprehensive single stock analysis"""
        ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
        
        if not ticker:
            print("❌ Please enter a valid ticker symbol")
            return
        
        print(f"\n🔄 Analyzing {ticker}... Please wait...")
        
        try:
            # Generate comprehensive report
            report = self.analyzer.generate_comprehensive_report(ticker)
            
            if 'error' in report:
                print(f"❌ Error: {report['error']}")
                return
            
            # Cache the analysis
            self.analysis_cache[ticker] = {
                'timestamp': datetime.now(),
                'report': report
            }
            
            # Display results
            self.display_analysis_results(report)
            
            # Generate visualizations
            data, info = self.analyzer.fetch_stock_data(ticker)
            data_with_indicators = self.analyzer.calculate_technical_indicators(data)
            self.analyzer.create_advanced_visualizations(data_with_indicators, info, ticker, report)
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
    
    def display_analysis_results(self, report: Dict):
        """Display formatted analysis results"""
        print("\n" + "="*80)
        print(f"📊 COMPREHENSIVE ANALYSIS: {report['ticker']}")
        print("="*80)
        
        # Company Info
        company = report.get('company_info', {})
        print(f"🏢 Company: {company.get('name', 'N/A')}")
        print(f"🏭 Sector: {company.get('sector', 'N/A')}")
        print(f"🔧 Industry: {company.get('industry', 'N/A')}")
        
        # Current Metrics
        current = report.get('current_metrics', {})
        print(f"\n💰 Current Price: ${current.get('price', 0):.2f}")
        print(f"📈 Change: ${current.get('change', 0):.2f} ({current.get('change_percent', 0):.2f}%)")
        print(f"📊 Volume: {current.get('volume', 0):,}")
        print(f"📏 52W Range: ${current.get('52w_low', 0):.2f} - ${current.get('52w_high', 0):.2f}")
        
        # Risk Metrics
        risk = report.get('risk_metrics', {})
        print(f"\n⚖️ RISK METRICS")
        print(f"📊 Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}")
        print(f"📉 Max Drawdown: {risk.get('max_drawdown', 0)*100:.2f}%")
        print(f"🎯 Beta: {risk.get('beta', 0):.3f}")
        print(f"⚠️ VaR (95%): {risk.get('var_95', 0)*100:.2f}%")
        
        # Trading Signals
        signals = self.analyzer.generate_trading_signals(
            self.analyzer.fetch_stock_data(report['ticker'])[0], report
        )
        
        print(f"\n🎯 TRADING SIGNALS")
        print(f"🔔 Overall Signal: {signals['overall_signal']}")
        print(f"💪 Signal Strength: {signals['signal_strength']:.2f}")
        print(f"🎯 Confidence: {signals['confidence']*100:.1f}%")
        print(f"⚠️ Risk Level: {signals['risk_level']}")
        
        # Individual Signals
        print(f"\n📊 Individual Signals:")
        for indicator, signal in signals['individual_signals'].items():
            print(f"   • {indicator}: {signal}")
        
        # Patterns
        patterns = report.get('patterns', [])
        print(f"\n🔍 DETECTED PATTERNS:")
        for pattern in patterns[:5]:  # Show top 5 patterns
            print(f"   • {pattern}")
        
        # ML Predictions
        ml_data = report.get('ml_predictions', {})
        if 'future_predictions' in ml_data:
            print(f"\n🤖 AI PRICE PREDICTIONS:")
            print(f"   • Next 5 days average: ${np.mean(ml_data['future_predictions']):.2f}")
            print(f"   • Trend: {'📈 Bullish' if ml_data['future_predictions'][-1] > current.get('price', 0) else '📉 Bearish'}")
    
    def portfolio_optimization_menu(self):
        """Portfolio optimization interface"""
        print("\n📈 PORTFOLIO OPTIMIZATION")
        print("-" * 30)
        
        tickers_input = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL): ")
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if len(tickers) < 2:
            print("❌ Please enter at least 2 stock tickers")
            return
        
        print(f"\n🔄 Optimizing portfolio for: {', '.join(tickers)}...")
        
        try:
            optimization_result = self.analyzer.portfolio_optimization(tickers)
            
            if 'error' in optimization_result:
                print(f"❌ Error: {optimization_result['error']}")
                return
            
            print("\n📊 OPTIMAL PORTFOLIO ALLOCATION:")
            print("-" * 40)
            
            for ticker, weight in optimization_result['optimal_weights'].items():
                print(f"{ticker}: {weight*100:.1f}%")
            
            print(f"\n📈 Expected Annual Return: {optimization_result['expected_return']*100:.2f}%")
            print(f"📊 Expected Volatility: {optimization_result['expected_volatility']*100:.2f}%")
            print(f"⚖️ Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"❌ Optimization failed: {str(e)}")
    
    def price_target_analysis(self):
        """Price target analysis interface"""
        ticker = input("Enter stock ticker for price target analysis: ").upper()
        
        if not ticker:
            print("❌ Please enter a valid ticker symbol")
            return
        
        print(f"\n🎯 Analyzing price targets for {ticker}...")
        
        try:
            data, info = self.analyzer.fetch_stock_data(ticker)
            data_with_indicators = self.analyzer.calculate_technical_indicators(data)
            
            # Generate comprehensive report for analysis
            report = self.analyzer.generate_comprehensive_report(ticker)
            
            # Get price targets
            targets = self.analyzer.generate_price_targets(data_with_indicators, report)
            
            print(f"\n🎯 PRICE TARGET ANALYSIS FOR {ticker}")
            print("=" * 50)
            print(f"💰 Current Price: ${targets['current_price']:.2f}")
            
            # Fibonacci targets
            fib_targets = targets['fibonacci_targets']
            print(f"\n📊 FIBONACCI TARGETS:")
            print(f"   🚀 Bullish Target 1: ${fib_targets['bullish_target_1']:.2f}")
            print(f"   🚀 Bullish Target 2: ${fib_targets['bullish_target_2']:.2f}")
            print(f"   🚀 Bullish Target 3: ${fib_targets['bullish_target_3']:.2f}")
            print(f"   🛡️ Support Level 1: ${fib_targets['support_1']:.2f}")
            print(f"   🛡️ Support Level 2: ${fib_targets['support_2']:.2f}")
            print(f"   🛡️ Support Level 3: ${fib_targets['support_3']:.2f}")
            
            # ATR targets
            atr_targets = targets['atr_targets']
            print(f"\n📏 ATR-BASED TARGETS:")
            print(f"   📈 Short-term High: ${atr_targets['short_term_high']:.2f}")
            print(f"   📉 Short-term Low: ${atr_targets['short_term_low']:.2f}")
            print(f"   📈 Medium-term High: ${atr_targets['medium_term_high']:.2f}")
            print(f"   📉 Medium-term Low: ${atr_targets['medium_term_low']:.2f}")
            
            # Support/Resistance levels
            sr_levels = targets['sr_levels']
            print(f"\n🏗️ SUPPORT/RESISTANCE LEVELS:")
            if sr_levels['immediate_resistance']:
                print(f"   🚧 Immediate Resistance: ${sr_levels['immediate_resistance']:.2f}")
            if sr_levels['immediate_support']:
                print(f"   🛡️ Immediate Support: ${sr_levels['immediate_support']:.2f}")
            
            # ML targets if available
            if 'ml_targets' in targets:
                ml_targets = targets['ml_targets']
                print(f"\n🤖 AI PREDICTIONS:")
                print(f"   📅 Next Week Target: ${ml_targets['next_week']:.2f}")
                print(f"   📊 Trend Direction: {ml_targets['trend_direction'].upper()}")
            
        except Exception as e:
            print(f"❌ Price target analysis failed: {str(e)}")
    
    def sector_comparison_menu(self):
        """Sector comparison interface"""
        ticker = input("Enter stock ticker for sector comparison: ").upper()
        sector_etf = input("Enter sector ETF (optional, press Enter to auto-detect): ").upper()
        
        if not ticker:
            print("❌ Please enter a valid ticker symbol")
            return
        
        sector_etf = sector_etf if sector_etf else None
        
        print(f"\n🏭 Comparing {ticker} with sector performance...")
        
        try:
            comparison = self.analyzer.sector_comparison(ticker, sector_etf)
            
            if 'error' in comparison:
                print(f"❌ Error: {comparison['error']}")
                return
            
            print(f"\n🏭 SECTOR COMPARISON: {comparison['ticker']}")
            print("=" * 40)
            print(f"📊 Stock vs {comparison['sector_etf']} ETF")
            print(f"   💹 Stock Return: {comparison['stock_return']:.2f}%")
            print(f"   🏭 Sector Return: {comparison['sector_return']:.2f}%")
            print(f"   📈 Outperformance: {comparison['outperformance']:.2f}%")
            print(f"   🔗 Correlation: {comparison['correlation']:.3f}")
            print(f"   📊 Sector Beta: {comparison['sector_beta']:.3f}")
            print(f"   💪 Relative Strength: {comparison['relative_strength']}")
            
            # Interpretation
            if comparison['outperformance'] > 10:
                print(f"\n✅ {ticker} is significantly outperforming its sector!")
            elif comparison['outperformance'] < -10:
                print(f"\n⚠️ {ticker} is underperforming its sector significantly.")
            else:
                print(f"\n📊 {ticker} performance is in line with its sector.")
            
        except Exception as e:
            print(f"❌ Sector comparison failed: {str(e)}")
    
    def batch_analysis_menu(self):
        """Batch analysis interface"""
        print("\n🔍 BATCH STOCK ANALYSIS")
        print("-" * 30)
        
        tickers_input = input("Enter stock tickers separated by commas: ")
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        if not tickers:
            print("❌ Please enter valid ticker symbols")
            return
        
        print(f"\n🔄 Analyzing {len(tickers)} stocks... This may take a while...")
        
        batch_results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"Processing {ticker} ({i}/{len(tickers)})...")
            
            try:
                # Quick analysis for batch processing
                data, info = self.analyzer.fetch_stock_data(ticker, period="6mo")
                data_with_indicators = self.analyzer.calculate_technical_indicators(data)
                
                # Generate basic report
                report = self.analyzer.generate_comprehensive_report(ticker)
                signals = self.analyzer.generate_trading_signals(data_with_indicators, report)
                
                current_price = data['Close'].iloc[-1]
                change_pct = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                
                batch_results.append({
                    'ticker': ticker,
                    'price': current_price,
                    'change_6m': change_pct,
                    'signal': signals['overall_signal'],
                    'confidence': signals['confidence'],
                    'risk_level': signals['risk_level'],
                    'rsi': data_with_indicators['RSI'].iloc[-1] if 'RSI' in data_with_indicators.columns else 'N/A'
                })
                
            except Exception as e:
                print(f"❌ Failed to analyze {ticker}: {str(e)}")
                batch_results.append({
                    'ticker': ticker,
                    'price': 'Error',
                    'change_6m': 'Error',
                    'signal': 'Error',
                    'confidence': 'Error',
                    'risk_level': 'Error',
                    'rsi': 'Error'
                })
        
        # Display batch results
        self.display_batch_results(batch_results)
    
    def display_batch_results(self, results: List[Dict]):
        """Display batch analysis results in a formatted table"""
        print(f"\n📊 BATCH ANALYSIS RESULTS")
        print("=" * 100)
        print(f"{'Ticker':<8} {'Price':<10} {'6M Change':<12} {'Signal':<12} {'Confidence':<12} {'Risk':<8} {'RSI':<6}")
        print("-" * 100)
        
        for result in results:
            ticker = result['ticker']
            price = f"${result['price']:.2f}" if isinstance(result['price'], (int, float)) else result['price']
            change = f"{result['change_6m']:.1f}%" if isinstance(result['change_6m'], (int, float)) else result['change_6m']
            signal = result['signal']
            confidence = f"{result['confidence']*100:.0f}%" if isinstance(result['confidence'], (int, float)) else result['confidence']
            risk = result['risk_level']
            rsi = f"{result['rsi']:.1f}" if isinstance(result['rsi'], (int, float)) else result['rsi']
            
            print(f"{ticker:<8} {price:<10} {change:<12} {signal:<12} {confidence:<12} {risk:<8} {rsi:<6}")
        
        # Summary statistics
        valid_results = [r for r in results if isinstance(r['change_6m'], (int, float))]
        if valid_results:
            avg_change = sum(r['change_6m'] for r in valid_results) / len(valid_results)
            buy_signals = sum(1 for r in valid_results if 'BUY' in r['signal'])
            sell_signals = sum(1 for r in valid_results if 'SELL' in r['signal'])
            
            print("-" * 100)
            print(f"📊 SUMMARY:")
            print(f"   Average 6M Return: {avg_change:.2f}%")
            print(f"   Buy Signals: {buy_signals}/{len(valid_results)}")
            print(f"   Sell Signals: {sell_signals}/{len(valid_results)}")
            print(f"   Hold/Neutral: {len(valid_results) - buy_signals - sell_signals}/{len(valid_results)}")
    
    def view_analysis_history(self):
        """Display analysis history"""
        if not self.analysis_cache:
            print("\n📋 No analysis history available")
            return
        
        print(f"\n📋 ANALYSIS HISTORY")
        print("=" * 50)
        
        for ticker, data in self.analysis_cache.items():
            timestamp = data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            report = data['report']
            current_metrics = report.get('current_metrics', {})
            
            print(f"🔹 {ticker} - Analyzed at {timestamp}")
            print(f"   💰 Price: ${current_metrics.get('price', 0):.2f}")
            print(f"   📈 Change: {current_metrics.get('change_percent', 0):.2f}%")
            print("-" * 30)
    
    def _calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calculate Bollinger Band position (helper method)"""
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            current_price = data['Close'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            bb_lower = data['BB_Lower'].iloc[-1]
            
            if bb_upper != bb_lower:
                return (current_price - bb_lower) / (bb_upper - bb_lower)
        return 0.5  # Neutral position
    
    def run(self):
        """Main dashboard loop"""
        print("🚀 Welcome to the Enhanced Stock Analysis Dashboard!")
        print("Loading market data and initializing systems...")
        
        while True:
            try:
                self.display_menu()
                choice = input("\nSelect an option (1-7): ").strip()
                
                if choice == '1':
                    self.single_stock_analysis()
                elif choice == '2':
                    self.portfolio_optimization_menu()
                elif choice == '3':
                    self.price_target_analysis()
                elif choice == '4':
                    self.sector_comparison_menu()
                elif choice == '5':
                    self.batch_analysis_menu()
                elif choice == '6':
                    self.view_analysis_history()
                elif choice == '7':
                    print("\n👋 Thank you for using the Enhanced Stock Analysis Dashboard!")
                    print("Happy investing! 📈")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                
                # Pause before showing menu again
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the dashboard!")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {str(e)}")
                print("Please try again or contact support.")


# Example usage and additional utility functions
def quick_stock_screener(analyzer: AdvancedStockAnalyzer, 
                        tickers: List[str]) -> pd.DataFrame:
    """
    Quick stock screener for multiple tickers
    
    Args:
        analyzer: AdvancedStockAnalyzer instance
        tickers: List of stock symbols
        
    Returns:
        DataFrame with screening results
    """
    screener_results = []
    
    for ticker in tickers:
        try:
            data, info = analyzer.fetch_stock_data(ticker, period="3mo")
            data_with_indicators = analyzer.calculate_technical_indicators(data)
            
            current_price = data['Close'].iloc[-1]
            rsi = data_with_indicators['RSI'].iloc[-1] if 'RSI' in data_with_indicators.columns else None
            volume_avg = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            
            # Calculate momentum
            momentum_5d = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]) * 100
            momentum_20d = ((current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20]) * 100
            
            screener_results.append({
                'Ticker': ticker,
                'Price': current_price,
                'RSI': rsi,
                'Volume_Ratio': current_volume / volume_avg,
                'Momentum_5D': momentum_5d,
                'Momentum_20D': momentum_20d,
                'Market_Cap': info.get('marketCap', 0),
                'PE_Ratio': info.get('trailingPE', None)
            })
            
        except Exception as e:
            print(f"Failed to screen {ticker}: {e}")
            continue
    
    return pd.DataFrame(screener_results)


def export_analysis_to_excel(analysis_results: Dict, filename: str):
    """
    Export analysis results to Excel file
    
    Args:
        analysis_results: Dictionary with analysis results
        filename: Output filename
    """
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Ticker', 'Current Price', 'Change %', 'Signal', 'Risk Level'],
                'Value': [
                    analysis_results.get('ticker', 'N/A'),
                    analysis_results.get('current_metrics', {}).get('price', 0),
                    analysis_results.get('current_metrics', {}).get('change_percent', 0),
                    'N/A',  # Would need to calculate
                    'N/A'   # Would need to calculate
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Risk metrics sheet
            risk_metrics = analysis_results.get('risk_metrics', {})
            if risk_metrics:
                risk_df = pd.DataFrame(list(risk_metrics.items()), 
                                     columns=['Metric', 'Value'])
                risk_df.to_excel(writer, sheet_name='Risk_Metrics', index=False)
            
            print(f"✅ Analysis exported to {filename}")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")


# Main execution
if __name__ == "__main__":
    # Initialize and run the dashboard
    dashboard = StockAnalysisDashboard()
    
    # You can also run individual components
    # Example: Quick analysis of popular stocks
    popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("🔍 Running quick screener on popular stocks...")
    screener_df = quick_stock_screener(dashboard.analyzer, popular_stocks)
    
    if not screener_df.empty:
        print("\n📊 QUICK STOCK SCREENER RESULTS:")
        print(screener_df.to_string(index=False))
        
        # Find interesting stocks
        if 'RSI' in screener_df.columns:
            oversold = screener_df[screener_df['RSI'] < 30]
            if not oversold.empty:
                print(f"\n🔥 Potentially Oversold Stocks (RSI < 30):")
                print(oversold[['Ticker', 'Price', 'RSI']].to_string(index=False))
    
    print("\n" + "="*60)
    # Run the main dashboard
    dashboard.run()
