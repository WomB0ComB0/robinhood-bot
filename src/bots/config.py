#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum, auto
import numpy as np
from datetime import timedelta


class OrderType(Enum):
    BUY_RECOMMENDATION = auto()
    SELL_RECOMMENDATION = auto()
    HOLD_RECOMMENDATION = auto()


class TimeFrame(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class StrategyType(Enum):
    SMA_CROSSOVER = "sma_crossover"
    VWAP = "vwap"
    RSI = "rsi"
    MACD = "macd"
    SENTIMENT = "sentiment"
    MOMENTUM = "momentum"


@dataclass
class TechnicalIndicators:
    """Configuration for technical indicators"""

    # SMA Configuration
    sma_short_period: int = 180
    sma_long_period: int = 365

    # VWAP Configuration
    vwap_period_days: int = 20
    vwap_threshold: float = 0.02

    # RSI Configuration
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD Configuration
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    # Momentum Configuration
    momentum_lookback_period: int = 14
    momentum_threshold: float = 0.0

    def __post_init__(self):
        self._validate_indicators()

    def _validate_indicators(self):
        if self.sma_short_period >= self.sma_long_period:
            raise ValueError("Short SMA period must be less than long SMA period")
        if self.vwap_period_days <= 0:
            raise ValueError("VWAP period must be positive")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ValueError("Invalid RSI thresholds")
        if self.macd_fast_period >= self.macd_slow_period:
            raise ValueError("MACD fast period must be less than slow period")


@dataclass
class RiskManagement:
    """Configuration for risk management"""

    # Position Sizing
    min_trade_amount: float = 5.00
    max_trade_amount: float = 100.00
    max_portfolio_per_stock: float = 0.20
    max_sector_allocation: float = 0.40

    # Stop Loss and Take Profit
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    stop_loss_percentage: float = 0.07
    take_profit_percentage: float = 0.15

    # Risk Metrics
    max_drawdown_percentage: float = 0.25
    max_daily_loss: float = 0.03
    max_weekly_loss: float = 0.10
    risk_reward_ratio: float = 2.0
    position_sizing_kelly: bool = True

    enable_trailing_stop: bool = False
    trailing_stop_percentage: float = 0.05

    def __post_init__(self):
        self._validate_risk_parameters()

    def _validate_risk_parameters(self):
        if not 0 < self.min_trade_amount < self.max_trade_amount:
            raise ValueError("Invalid trade amount limits")
        if not 0 < self.max_portfolio_per_stock < 1:
            raise ValueError("Invalid portfolio allocation per stock")
        if not 0 < self.max_sector_allocation < 1:
            raise ValueError("Invalid sector allocation")
        if not 0 < self.stop_loss_percentage < self.take_profit_percentage < 1:
            raise ValueError("Invalid stop loss or take profit percentages")


@dataclass
class SentimentAnalysis:
    """Configuration for sentiment analysis"""

    enable_sentiment: bool = False
    sentiment_threshold_buy: float = 0.15
    sentiment_threshold_sell: float = -0.15
    max_tweets_analyze: int = 100
    sentiment_lookback_days: int = 7
    min_sentiment_samples: int = 30
    sentiment_weight: float = 0.3

    def __post_init__(self):
        self._validate_sentiment_parameters()

    def _validate_sentiment_parameters(self):
        if not -1 <= self.sentiment_threshold_sell < self.sentiment_threshold_buy <= 1:
            raise ValueError("Invalid sentiment thresholds")
        if self.max_tweets_analyze < self.min_sentiment_samples:
            raise ValueError("Max tweets must be greater than minimum samples")


@dataclass
class TradingConfig:
    """Configuration for trading"""

    risk_management: RiskManagement = field(default_factory=RiskManagement)
    technical_indicators: TechnicalIndicators = field(default_factory=TechnicalIndicators)
    sentiment_analysis: SentimentAnalysis = field(default_factory=SentimentAnalysis)

    trade_interval_hours: int = 4
    trading_timeframe: TimeFrame = TimeFrame.HOUR_4
    market_hours_only: bool = True

    enabled_strategies: List[StrategyType] = field(
        default_factory=lambda: [StrategyType.SMA_CROSSOVER, StrategyType.VWAP, StrategyType.RSI]
    )
    strategy_weights: Dict[StrategyType, float] = field(
        default_factory=lambda: {StrategyType.SMA_CROSSOVER: 0.4, StrategyType.VWAP: 0.3, StrategyType.RSI: 0.3}
    )

    enable_dynamic_threshold: bool = True
    threshold_percentage: float = 0.01
    volatility_adjustment: bool = True
    market_regime_aware: bool = True
    trade_threshold: float = 0.015

    def __post_init__(self):
        self._validate_config()

    def _validate_config(self):
        """Validate the complete trading configuration."""
        if self.trade_interval_hours <= 0:
            raise ValueError("Trade interval must be positive")

        if sum(self.strategy_weights.values()) != 1.0:
            raise ValueError("Strategy weights must sum to 1.0")

        for strategy in self.enabled_strategies:
            if strategy not in self.strategy_weights:
                raise ValueError(f"Missing weight for strategy {strategy}")

    def get_trading_interval(self) -> timedelta:
        """Get trading interval as timedelta."""
        return timedelta(hours=self.trade_interval_hours)

    def adjust_thresholds_for_volatility(self, historical_volatility: float) -> None:
        """Adjust thresholds based on market volatility."""
        if self.volatility_adjustment:
            volatility_factor = np.clip(historical_volatility / 0.20, 0.5, 2.0)
            self.risk_management.stop_loss_percentage *= volatility_factor
            self.risk_management.take_profit_percentage *= volatility_factor
            self.technical_indicators.vwap_threshold *= volatility_factor

    def get_position_size(self, account_value: float) -> float:
        """Calculate position size using Kelly Criterion if enabled."""
        if not self.risk_management.position_sizing_kelly:
            return min(
                account_value * self.risk_management.max_portfolio_per_stock, self.risk_management.max_trade_amount
            )

        win_rate = 0.55
        risk_ratio = self.risk_management.risk_reward_ratio
        kelly_percentage = (win_rate * risk_ratio - (1 - win_rate)) / risk_ratio
        kelly_percentage = min(kelly_percentage, self.risk_management.max_portfolio_per_stock)

        return min(account_value * kelly_percentage, self.risk_management.max_trade_amount)

    def should_trade_now(self, current_time) -> bool:
        """Determine if trading should occur at the current time."""
        if self.market_hours_only:
            market_open = current_time.replace(hour=9, minute=30)
            market_close = current_time.replace(hour=16, minute=0)
            return market_open <= current_time <= market_close
        return True

    def get_strategy_decision(self, strategy_signals: Dict[StrategyType, float]) -> float:
        """Calculate weighted strategy decision."""
        final_signal = 0.0
        for strategy, signal in strategy_signals.items():
            if strategy in self.enabled_strategies:
                final_signal += signal * self.strategy_weights.get(strategy, 0)
        return final_signal
