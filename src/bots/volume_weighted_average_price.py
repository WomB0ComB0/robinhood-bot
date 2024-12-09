#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
from src.bots.config import OrderType
import robin_stocks.robinhood as robinhood


@dataclass
class VWAPMetrics:
    """Container for VWAP-related metrics"""

    vwap: float
    std_dev: float
    upper_band: float
    lower_band: float
    volume_ratio: float


@dataclass
class PortfolioMetrics:
    total_equity: float
    cash_available: float
    positions: Dict[str, float]
    sector_exposure: Dict[str, float]
    beta_weighted_delta: float
    sharpe_ratio: float
    sortino_ratio: float


class PortfolioManager:
    def __init__(self, initial_capital: float, max_position_size: float = 0.2):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.positions: Dict[str, Dict] = {}
        self.sector_mappings: Dict[str, str] = self._initialize_sector_mappings()

    def calculate_position_size(self, ticker: str, signal_strength: float, volatility: float) -> float:
        """Calculate optimal position size using Kelly Criterion and volatility adjustment"""
        portfolio_metrics = self.get_portfolio_metrics()
        available_capital = portfolio_metrics.cash_available

        # Kelly Criterion calculation
        win_rate = 0.55  # Historical win rate
        win_loss_ratio = 1.5  # Historical profit/loss ratio
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Volatility adjustment
        vol_factor = np.exp(-volatility)

        # Position size calculation
        base_size = available_capital * kelly_percentage * vol_factor
        adjusted_size = base_size * abs(signal_strength)

        # Apply position limits
        max_allowed = available_capital * self.max_position_size
        return min(adjusted_size, max_allowed)

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        total_equity = sum(pos["market_value"] for pos in self.positions.values())
        sector_exposure = self._calculate_sector_exposure()

        returns = self._calculate_historical_returns()
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)

        return PortfolioMetrics(
            total_equity=total_equity,
            cash_available=self.initial_capital - total_equity,
            positions=self.positions,
            sector_exposure=sector_exposure,
            beta_weighted_delta=self._calculate_portfolio_delta(),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
        )

    def _calculate_portfolio_delta(self) -> float:
        """Calculate portfolio delta for risk management"""
        total_delta = 0.0
        for ticker, position in self.positions.items():
            beta = self._get_beta(ticker)
            delta = position["market_value"] * beta
            total_delta += delta
        return total_delta


class TradeBotVWAP:
    """
    Trading bot implementing Volume-Weighted Average Price (VWAP) strategy with enhanced features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VWAP trading bot with configuration."""
        self.config = config or {
            "enable_dynamic_threshold": True,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.03,
            "vwap_window": 20,
            "volume_threshold": 1.5,
            "max_position_size": 100000,  # Maximum position size in dollars
            "risk_per_trade": 0.02,  # Maximum risk per trade (2%)
            "trailing_stop": 0.015,  # Trailing stop distance (1.5%)
            "mean_reversion_threshold": 2.0,  # Standard deviations for mean reversion
        }

        self.logger = self._setup_logging()
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit_loss": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "average_profit_per_trade": 0.0,
            "sharpe_ratio": 0.0,
        }

        self.trade_history: List[Dict[str, Any]] = []
        self.highest_equity = 0.0

    def _setup_logging(self) -> logging.Logger:
        """Configure logging with enhanced formatting and rotation."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def get_stock_history_dataframe(self, ticker: str, interval: str, span: str) -> pd.DataFrame:
        """Fetch historical stock data from Robinhood and return as a DataFrame."""
        try:
            # Use 'year' span instead of 'day' to get more data points
            historicals = robinhood.stocks.get_stock_historicals(
                ticker, interval=interval, span="year", bounds="regular"  # Changed from 'day' to 'year'
            )

            if not historicals:
                self.logger.warning("No historical data available for %s", ticker)
                return pd.DataFrame()

            df = pd.DataFrame(historicals)
            df["begins_at"] = pd.to_datetime(df["begins_at"])

            # Convert price columns to float
            for col in ["open_price", "close_price", "high_price", "low_price"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            return df

        except (Exception, pd.errors.EmptyDataError) as e:
            self.logger.error("Error fetching historical data for %s: %s", ticker, str(e))
            return pd.DataFrame()

    def calculate_vwap_metrics(self, df: pd.DataFrame) -> VWAPMetrics:
        """Calculate VWAP and related metrics from price and volume data."""
        try:
            df = df.copy()

            for col in ["close_price", "volume", "high", "low"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["typical_price"] = (df["high"] + df["low"] + df["close_price"]) / 3

            df["cum_vol"] = df["volume"].cumsum()
            df["cum_vol_price"] = (df["typical_price"] * df["volume"]).cumsum()

            df["vwap"] = df["cum_vol_price"] / df["cum_vol"]

            window = self.config["vwap_window"]
            df["vwap_std"] = df["vwap"].rolling(window=window).std()
            df["volume_sma"] = df["volume"].rolling(window=window).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

            latest = df.iloc[-1]

            return VWAPMetrics(
                vwap=latest["vwap"],
                std_dev=latest["vwap_std"],
                upper_band=latest["vwap"] + (2 * latest["vwap_std"]),
                lower_band=latest["vwap"] - (2 * latest["vwap_std"]),
                volume_ratio=latest["volume_ratio"],
            )

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error calculating VWAP metrics: %s", str(e))
            raise ValueError("VWAP calculation failed") from e

    def make_trading_decision(
        self, ticker: str, current_price: float, position: Optional[Dict[str, Any]] = None
    ) -> OrderType:
        """Make trading decision based on VWAP analysis and risk management."""
        try:
            # Get both intraday and daily data
            intraday_df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")
            daily_df = self.get_stock_history_dataframe(ticker, interval="day", span="year")

            intraday_metrics = self.calculate_vwap_metrics(intraday_df)
            daily_metrics = self.calculate_vwap_metrics(daily_df)

            if position:
                if self._check_risk_management(current_price, position):
                    return OrderType.SELL_RECOMMENDATION

            threshold = (
                intraday_metrics.std_dev / intraday_metrics.vwap if self.config["enable_dynamic_threshold"] else 0.01
            )

            price_to_vwap = (current_price - intraday_metrics.vwap) / intraday_metrics.vwap

            sufficient_volume = intraday_metrics.volume_ratio > self.config["volume_threshold"]

            mean_reversion_threshold = self.config["mean_reversion_threshold"]

            if (
                price_to_vwap < -threshold
                and current_price < daily_metrics.lower_band
                and sufficient_volume
                and abs(price_to_vwap) < mean_reversion_threshold
            ):
                return OrderType.BUY_RECOMMENDATION

            elif (
                price_to_vwap > threshold
                and current_price > daily_metrics.upper_band
                and abs(price_to_vwap) > mean_reversion_threshold
            ):
                return OrderType.SELL_RECOMMENDATION

            return OrderType.HOLD_RECOMMENDATION

        except (Exception, pd.errors.EmptyDataError) as e:
            self.logger.error("Error making trading decision: %s", str(e))
            return OrderType.HOLD_RECOMMENDATION

    def _check_risk_management(self, current_price: float, position: Dict[str, Any]) -> bool:
        """Enhanced risk management with trailing stops and volatility-adjusted thresholds."""
        try:
            avg_price = float(position["average_buy_price"])
            quantity = float(position["quantity"])
            position_value = current_price * quantity

            unrealized_pl = (current_price - avg_price) / avg_price

            position["highest_value"] = max(position.get("highest_value", position_value), position_value)

            if unrealized_pl < -self.config["stop_loss_percentage"]:
                self.logger.info("Stop loss triggered at %s", unrealized_pl)
                return True

            if unrealized_pl > self.config["take_profit_percentage"]:
                self.logger.info("Take profit triggered at %s", unrealized_pl)
                return True

            trailing_stop_distance = self.config["trailing_stop"]
            if (position["highest_value"] - position_value) / position["highest_value"] > trailing_stop_distance:
                self.logger.info("Trailing stop triggered at %s", position_value / position["highest_value"])
                return True

            if position_value > self.config["max_position_size"]:
                self.logger.info("Maximum position size exceeded")
                return True

            return False

        except (Exception, pd.errors.EmptyDataError) as e:
            self.logger.error("Error in risk management: %s", str(e))
            return False

    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update performance tracking metrics after each trade."""
        try:
            self.performance_metrics["total_trades"] += 1

            profit_loss = trade_result["profit_loss"]

            if profit_loss > 0:
                self.performance_metrics["successful_trades"] += 1
            else:
                self.performance_metrics["failed_trades"] += 1

            self.performance_metrics["total_profit_loss"] += profit_loss

            self.trade_history.append(
                {
                    "timestamp": datetime.now(),
                    "profit_loss": profit_loss,
                    "type": trade_result.get("type", "unknown"),
                    "price": trade_result.get("price", 0.0),
                    "quantity": trade_result.get("quantity", 0.0),
                }
            )

            self._update_advanced_metrics()

        except (Exception, pd.errors.EmptyDataError) as e:
            self.logger.error("Error updating performance metrics: %s", str(e))

    def _update_advanced_metrics(self):
        """Update advanced performance metrics."""
        try:
            total_trades = self.performance_metrics["total_trades"]
            if total_trades == 0:
                return

            self.performance_metrics["win_rate"] = self.performance_metrics["successful_trades"] / total_trades

            self.performance_metrics["average_profit_per_trade"] = (
                self.performance_metrics["total_profit_loss"] / total_trades
            )

            equity_curve = []
            running_total = 0
            peak = 0
            max_drawdown = 0

            for trade in self.trade_history:
                running_total += trade["profit_loss"]
                equity_curve.append(running_total)
                peak = max(peak, running_total)
                drawdown = (peak - running_total) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)

            self.performance_metrics["max_drawdown"] = max_drawdown

            if len(equity_curve) > 1:
                returns = pd.Series(equity_curve).pct_change().dropna()
                excess_returns = returns - 0.02 / 252
                sharpe_ratio = (
                    np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
                )
                self.performance_metrics["sharpe_ratio"] = sharpe_ratio

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error updating advanced metrics: %s", str(e))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of trading performance."""
        return {
            "total_trades": self.performance_metrics["total_trades"],
            "win_rate": f"{self.performance_metrics['win_rate']:.2%}",
            "total_profit_loss": f"${self.performance_metrics['total_profit_loss']:,.2f}",
            "average_profit_per_trade": f"${self.performance_metrics['average_profit_per_trade']:,.2f}",
            "max_drawdown": f"{self.performance_metrics['max_drawdown']:.2%}",
            "sharpe_ratio": f"{self.performance_metrics['sharpe_ratio']:.2f}",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


class PerformanceAnalyzer:
    def __init__(self):
        self.trade_history: List[Dict] = []
        self.daily_returns: pd.Series = pd.Series()
        self.benchmark_returns: pd.Series = pd.Series()

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Record a new trade with enhanced metrics"""
        self.trade_history.append(
            {
                **trade,
                "timestamp": datetime.now(),
                "market_conditions": self._get_market_conditions(),
                "trade_metrics": self._calculate_trade_metrics(trade),
            }
        )
        self._update_performance_metrics()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.trade_history:
            return self._empty_performance_report()

        metrics = {
            "total_return": self._calculate_total_return(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "max_drawdown": self._calculate_max_drawdown(),
            "win_rate": self._calculate_win_rate(),
            "profit_factor": self._calculate_profit_factor(),
            "risk_adjusted_return": self._calculate_risk_adjusted_return(),
            "alpha": self._calculate_alpha(),
            "beta": self._calculate_beta(),
            "information_ratio": self._calculate_information_ratio(),
        }

        return {
            **metrics,
            "daily_stats": self._calculate_daily_statistics(),
            "monthly_stats": self._calculate_monthly_statistics(),
            "drawdown_analysis": self._analyze_drawdowns(),
            "risk_metrics": self._calculate_risk_metrics(),
        }
