#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from src.bots.base_trade_bot import OrderType, TradeBot, OrderResult
from src.bots.config import TradingConfig, StrategyType
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TradeBotSimpleMovingAverage(TradeBot):
    """
    Enhanced Simple Moving Average trading bot that implements a sophisticated SMA strategy
    with dynamic thresholds, trend analysis, and risk management.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize the SMA trading bot with configuration.

        Args:
            config: Optional trading configuration. If None, uses default config.
        """
        super().__init__(config=config if config else TradingConfig())
        self.strategy_type = StrategyType.SMA_CROSSOVER
        self._validate_sma_config()

    def _validate_sma_config(self) -> None:
        """Validate SMA-specific configuration parameters."""
        if self.config.technical_indicators.sma_short_period >= self.config.technical_indicators.sma_long_period:
            raise ValueError("Short period must be less than long period")

        if self.config.technical_indicators.momentum_lookback_period <= 0:
            raise ValueError("Momentum lookback period must be positive")

    def calculate_technical_indicators(self, df: pd.DataFrame, period: int) -> Dict[str, float]:
        """
        Calculate technical indicators for the given period.

        Args:
            df: DataFrame containing price data
            period: Rolling window period for calculations

        Returns:
            Dictionary containing calculated technical indicators
        """
        if df is None or df.empty or not period:
            logger.warning("Invalid data provided for technical indicator calculation")
            return {"sma": 0.0, "std": 0.0, "momentum": 0.0, "volatility": 0.0}

        try:
            df = df.copy()
            df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")

            # Handle NaN values
            df = df.dropna(subset=["close_price"])

            if len(df) < period:
                logger.warning("Insufficient data points for period %s", period)
                return {"sma": 0.0, "std": 0.0, "momentum": 0.0, "volatility": 0.0}

            # Calculate SMA and Standard Deviation
            df["SMA"] = df["close_price"].rolling(window=period, min_periods=1).mean()
            df["STD"] = df["close_price"].rolling(window=period, min_periods=1).std()

            # Calculate Momentum (rate of change)
            df["momentum"] = df["close_price"].pct_change(self.config.technical_indicators.momentum_lookback_period)

            # Calculate Historical Volatility
            df["log_return"] = np.log(df["close_price"] / df["close_price"].shift(1))
            df["volatility"] = df["log_return"].rolling(window=period).std() * np.sqrt(252)

            return {
                "sma": round(df["SMA"].iloc[-1], 4),
                "std": round(df["STD"].iloc[-1], 4),
                "momentum": round(df["momentum"].iloc[-1], 4),
                "volatility": round(df["volatility"].iloc[-1], 4),
            }

        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            logger.error("Error calculating technical indicators: %s", str(e))
            return {"sma": 0.0, "std": 0.0, "momentum": 0.0, "volatility": 0.0}

    def analyze_market_conditions(
        self, short_term: Dict[str, float], long_term: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Analyze market conditions using technical indicators.

        Args:
            short_term: Short-term technical indicators
            long_term: Long-term technical indicators

        Returns:
            Tuple of (trend_strength, signal_strength, momentum_signal)
        """
        # Calculate trend strength using multiple factors
        price_trend = (short_term["sma"] - long_term["sma"]) / long_term["sma"]
        volatility_ratio = short_term["volatility"] / long_term["volatility"] if long_term["volatility"] > 0 else 1.0

        trend_strength = abs(price_trend) * (1 + volatility_ratio)

        signal_strength = price_trend * (1 + abs(short_term["momentum"]))

        momentum_signal = short_term["momentum"] > self.config.technical_indicators.momentum_threshold

        return trend_strength, signal_strength, momentum_signal

    def calculate_position_size(self, ticker: str, signal_strength: float) -> float:
        """
        Calculate position size based on signal strength and risk parameters.

        Args:
            ticker: Stock ticker symbol
            signal_strength: Strength of the trading signal

        Returns:
            Position size in dollars
        """
        account_value = self.get_current_cash_position()
        base_position = self.config.get_position_size(account_value, self._calculate_volatility(ticker))

        confidence_factor = min(abs(signal_strength), 1.0)
        adjusted_position = base_position * confidence_factor

        return max(
            min(adjusted_position, self.config.risk_management.max_trade_amount),
            self.config.risk_management.min_trade_amount,
        )

    def make_order_recommendation(self, ticker: str) -> Optional[OrderType]:
        """
        Generate trading signals based on SMA strategy and market conditions.

        Args:
            ticker: Stock ticker symbol

        Returns:
            OrderType recommendation
        """
        try:
            if not ticker:
                return None

            # Get historical data and current market conditions
            stock_history_df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")
            current_price = self.get_current_market_price(ticker)
            position = self.get_current_positions().get(ticker)

            if position and self.check_risk_management(current_price, position):
                logger.info("Risk management triggered for %s", ticker)
                return OrderType.SELL_RECOMMENDATION

            short_term = self.calculate_technical_indicators(
                stock_history_df, self.config.technical_indicators.sma_short_period
            )
            long_term = self.calculate_technical_indicators(
                stock_history_df, self.config.technical_indicators.sma_long_period
            )

            trend_strength, signal_strength, momentum_signal = self.analyze_market_conditions(short_term, long_term)

            threshold = (
                self.config.threshold_percentage * (1 + trend_strength)
                if self.config.enable_dynamic_threshold
                else self.config.threshold_percentage
            )

            if signal_strength > threshold and momentum_signal:
                return OrderType.BUY_RECOMMENDATION
            elif signal_strength < -threshold or (signal_strength < 0 and not momentum_signal):
                return OrderType.SELL_RECOMMENDATION

            return OrderType.HOLD_RECOMMENDATION

        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            logger.error("Error generating order recommendation: %s", str(e))
            return OrderType.HOLD_RECOMMENDATION

    def get_current_positions(self) -> Dict[str, Dict[str, float]]:
        """Retrieve current positions with SMA-specific position metrics."""
        try:
            # Get base positions from parent class
            base_positions = super().get_current_positions()
            enhanced_positions = {}

            for ticker, position in base_positions.items():
                # Get historical data for SMA calculations
                df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")

                if df.empty:
                    continue

                # Calculate additional SMA-specific metrics
                short_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_short_period)
                long_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_long_period)

                # Enhance position data with SMA metrics
                enhanced_positions[ticker] = {
                    **position,  # Include all base position data
                    "short_term_sma": short_term["sma"],
                    "long_term_sma": long_term["sma"],
                    "current_momentum": short_term["momentum"],
                    "current_volatility": short_term["volatility"],
                    "trend_strength": self.analyze_market_conditions(short_term, long_term)[0],
                }

            return enhanced_positions

        except (KeyError, ValueError) as e:
            logger.error("Error retrieving enhanced positions: %s", str(e))
            return super().get_current_positions()

    def execute_trade(self, ticker: str) -> OrderResult:
        """
        Execute trade based on strategy recommendation and position sizing.

        Args:
            ticker: Stock ticker symbol

        Returns:
            OrderResult containing trade execution details
        """
        try:
            if not self.config.should_trade_now(datetime.now(timezone.utc)):
                return OrderResult(success=False, error_message="Outside trading hours", amount=0.0)

            recommendation = self.make_order_recommendation(ticker)

            if recommendation == OrderType.BUY_RECOMMENDATION:
                position_size = self.calculate_position_size(
                    ticker,
                    self.analyze_market_conditions(
                        self.calculate_technical_indicators(
                            self.get_stock_history_dataframe(ticker, interval="5minute", span="day"),
                            self.config.technical_indicators.sma_short_period,
                        ),
                        self.calculate_technical_indicators(
                            self.get_stock_history_dataframe(ticker, interval="5minute", span="day"),
                            self.config.technical_indicators.sma_long_period,
                        ),
                    )[1],
                )
                return self.place_order(ticker, OrderType.BUY_RECOMMENDATION, position_size)

            elif recommendation == OrderType.SELL_RECOMMENDATION:
                position = self.get_current_positions().get(ticker)
                if position:
                    return self.place_order(ticker, OrderType.SELL_RECOMMENDATION, float(position["equity"]))

            return OrderResult(success=True, error_message="No trade conditions met", amount=0.0)

        except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
            logger.error("Error executing trade: %s", str(e))
            return OrderResult(success=False, error_message=str(e), amount=0.0)

    def get_stock_history_dataframe(self, ticker: str, interval: str = "1day", span: str = "year") -> pd.DataFrame:
        """Retrieve and preprocess historical stock data with SMA-specific enhancements."""
        try:
            # Get base historical data from parent class
            df = super().get_stock_history_dataframe(ticker, interval, span)

            if df.empty:
                return df

            # Add SMA-specific calculations
            short_period = self.config.technical_indicators.sma_short_period
            long_period = self.config.technical_indicators.sma_long_period

            # Calculate SMAs
            df["short_sma"] = df["close_price"].rolling(window=short_period, min_periods=1).mean()
            df["long_sma"] = df["close_price"].rolling(window=long_period, min_periods=1).mean()

            # Calculate crossover signals
            df["sma_crossover"] = np.where(
                df["short_sma"] > df["long_sma"],
                1,  # Bullish crossover
                np.where(df["short_sma"] < df["long_sma"], -1, 0),  # Bearish crossover  # No crossover
            )

            # Calculate momentum
            df["momentum"] = df["close_price"].pct_change(self.config.technical_indicators.momentum_lookback_period)

            # Calculate volatility
            df["log_returns"] = np.log(df["close_price"] / df["close_price"].shift(1))
            df["volatility"] = df["log_returns"].rolling(
                window=self.config.technical_indicators.sma_long_period
            ).std() * np.sqrt(252)

            return df

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error retrieving enhanced stock history: %s", str(e))
            return pd.DataFrame()

    def get_current_market_price(self, ticker: str) -> float:
        """Retrieve the current market price with SMA context."""
        try:
            current_price = super().get_current_market_price(ticker)

            if current_price <= 0:
                return 0.0

            # Get recent historical data for SMA context
            df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")

            if df.empty:
                return current_price

            # Calculate SMAs for validation
            short_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_short_period)
            long_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_long_period)

            # Validate price against SMAs
            if abs(current_price - short_term["sma"]) / short_term["sma"] > 0.1:
                logger.warning(
                    "Current price %s deviates significantly from short-term SMA %s",
                    current_price,
                    short_term["sma"],
                )

            return current_price

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error retrieving current market price: %s", str(e))
            return super().get_current_market_price(ticker)

    def check_risk_management(self, current_price: float, position: Dict[str, float]) -> bool:
        """Enhanced risk management check incorporating SMA-specific criteria."""
        try:
            # Check base risk management conditions
            if super().check_risk_management(current_price, position):
                return True

            # Get historical data for SMA-specific risk checks
            ticker = position.get("symbol")
            if not ticker:
                return False

            df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")

            if df.empty:
                return False

            # Calculate SMA-specific risk metrics
            short_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_short_period)
            long_term = self.calculate_technical_indicators(df, self.config.technical_indicators.sma_long_period)

            # Check for trend reversal
            trend_strength, signal_strength, momentum_signal = self.analyze_market_conditions(short_term, long_term)

            # Exit if strong trend reversal is detected
            if (
                trend_strength > self.config.threshold_percentage
                and signal_strength < -self.config.threshold_percentage
            ):
                logger.info("SMA trend reversal detected for %s", ticker)
                return True

            # Check volatility threshold
            if short_term["volatility"] > self.config.risk_management.max_volatility:
                logger.info("Volatility threshold exceeded for %s", ticker)
                return True

            # Check momentum breakdown
            if (
                position.get("current_momentum", 0) > 0
                and short_term["momentum"] < -self.config.technical_indicators.momentum_threshold
            ):
                logger.info("Momentum breakdown detected for %s", ticker)
                return True

            return False

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error in SMA risk management check: %s", str(e))
            return super().check_risk_management(current_price, position)

    def get_current_cash_position(self) -> float:
        """Retrieve current cash position with SMA-specific allocation logic."""
        try:
            base_cash = super().get_current_cash_position()

            # Adjust available cash based on market conditions
            active_positions = self.get_current_positions()

            # Calculate aggregate trend strength
            total_trend_strength = 0
            for position in active_positions.values():
                total_trend_strength += position.get("trend_strength", 0)

            # Adjust cash availability based on market conditions
            if len(active_positions) > 0:
                avg_trend_strength = total_trend_strength / len(active_positions)

                # Reduce available cash in strong trends to maintain positions
                if avg_trend_strength > self.config.threshold_percentage:
                    return base_cash * 0.8  # Reserve 20% for existing positions
                # Increase available cash in weak trends
                elif avg_trend_strength < 0:
                    return base_cash * 1.2  # Allow for opportunistic buying

            return base_cash

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error retrieving adjusted cash position: %s", str(e))
            return super().get_current_cash_position()

    def execute_buy_order(self, ticker, amount):
        """Implement the logic to execute a buy order."""
        try:
            # Check if there are sufficient funds
            if not self.has_sufficient_funds_available(amount):
                logger.error("Insufficient funds to execute buy order for %s", ticker)
                return OrderResult(success=False, error_message="Insufficient funds", amount=0.0)

            # Place the buy order
            order_result = self.place_order(ticker, OrderType.BUY_RECOMMENDATION, amount)
            logger.info("Buy order executed for %s with amount %s", ticker, amount)
            return order_result

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error executing buy order: %s", str(e))
            return OrderResult(success=False, error_message=str(e), amount=0.0)

    def execute_sell_order(self, ticker, amount):
        """Implement the logic to execute a sell order."""
        try:
            # Check if there is sufficient equity
            if not self.has_sufficient_equity(ticker, amount):
                logger.error("Insufficient equity to execute sell order for %s", ticker)
                return OrderResult(success=False, error_message="Insufficient equity", amount=0.0)

            # Place the sell order
            order_result = self.place_order(ticker, OrderType.SELL_RECOMMENDATION, amount)
            logger.info("Sell order executed for %s with amount %s", ticker, amount)
            return order_result

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error executing sell order: %s", str(e))
            return OrderResult(success=False, error_message=str(e), amount=0.0)
