#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging
from typing import Dict, Optional, List
import pandas as pd
import pyotp
import robin_stocks.robinhood as robinhood
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from src.utilities import RobinhoodCredentials
from src.bots.config import TradingConfig, StrategyType, OrderType

# Configure logging to write to a file
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trade_bot.log"),  # Log to a file named trade_bot.log
        logging.StreamHandler(),  # Also log to the console
    ],
)

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    details: Optional[Dict] = None
    amount: float = 0.0
    price: float = 0.0
    timestamp: datetime = datetime.now(timezone.utc)


class TradeBot:
    def __init__(self, config: TradingConfig):
        """Initialize TradeBot with enhanced configuration and safety checks."""
        self.config = config
        self.validate_config()
        self._authenticate()
        self.last_trade_time: Dict[str, datetime] = {}
        self.trade_history: List[OrderResult] = []

    def _authenticate(self) -> None:
        """Handle Robinhood authentication with improved error handling."""
        credentials = RobinhoodCredentials()
        totp = None

        logger.debug("Starting authentication process.")
        logger.debug("User: %s", credentials.user)
        logger.debug("MFA Code: %s", credentials.mfa_code)  # Ensure this is a valid base32 string

        if credentials.mfa_code:
            logger.debug("MFA code is provided, attempting to generate TOTP.")
            try:
                totp = pyotp.TOTP(credentials.mfa_code).now()
            except Exception as e:
                logger.error("Failed to generate MFA code: %s", e)
                logger.debug("MFA code provided: %s", credentials.mfa_code)
                logger.debug("Ensure the MFA code is a valid base32 string.")
                raise ValueError("Invalid MFA configuration") from e

        try:
            robinhood.login(credentials.user, credentials.password, mfa_code=totp, expiresIn=86400, by_sms=False)
            logger.info("Successfully authenticated with Robinhood")
        except Exception as e:
            logger.error("Authentication failed: %s", e)
            raise ConnectionError("Failed to connect to Robinhood: %s" % e) from e

    def calculate_strategy_signals(self, ticker: str) -> Dict[StrategyType, float]:
        """Calculate signals for all enabled strategies."""
        signals = {}
        df = self.get_stock_history_dataframe(ticker, interval="day", span="year")

        if df.empty:
            return {strategy: 0.0 for strategy in self.config.enabled_strategies}

        df["close"] = df["close_price"].astype(float)
        df["volume"] = df["volume"].astype(float)

        for strategy in self.config.enabled_strategies:
            if strategy == StrategyType.SMA_CROSSOVER:
                signals[strategy] = self._calculate_sma_signal(df)
            elif strategy == StrategyType.VWAP:
                signals[strategy] = self._calculate_vwap_signal(df)
            elif strategy == StrategyType.RSI:
                signals[strategy] = self._calculate_rsi_signal(df)
            elif strategy == StrategyType.MACD:
                signals[strategy] = self._calculate_macd_signal(df)

        return signals

    def _calculate_sma_signal(self, df: pd.DataFrame) -> float:
        """Calculate SMA crossover signal."""
        short_sma = df["close"].rolling(window=self.config.technical_indicators.sma_short_period).mean()
        long_sma = df["close"].rolling(window=self.config.technical_indicators.sma_long_period).mean()

        if short_sma.iloc[-1] > long_sma.iloc[-1]:
            return 1.0
        elif short_sma.iloc[-1] < long_sma.iloc[-1]:
            return -1.0
        return 0.0

    def _calculate_vwap_signal(self, df: pd.DataFrame) -> float:
        """Calculate VWAP signal."""
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        current_price = df["close"].iloc[-1]
        vwap = df["vwap"].iloc[-1]

        threshold = self.config.technical_indicators.vwap_threshold
        if current_price > vwap * (1 + threshold):
            return -1.0
        elif current_price < vwap * (1 - threshold):
            return 1.0
        return 0.0

    def _calculate_rsi_signal(self, df: pd.DataFrame) -> float:
        """Calculate RSI signal."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.technical_indicators.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.technical_indicators.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        if current_rsi < self.config.technical_indicators.rsi_oversold:
            return 1.0
        elif current_rsi > self.config.technical_indicators.rsi_overbought:
            return -1.0
        return 0.0

    def _calculate_macd_signal(self, df: pd.DataFrame) -> float:
        """Calculate MACD signal."""
        short_ema = df["close"].ewm(span=self.config.technical_indicators.macd_short_period, adjust=False).mean()
        long_ema = df["close"].ewm(span=self.config.technical_indicators.macd_long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=self.config.technical_indicators.macd_signal_period, adjust=False).mean()

        if macd.iloc[-1] > signal_line.iloc[-1]:
            return 1.0
        elif macd.iloc[-1] < signal_line.iloc[-1]:
            return -1.0
        return 0.0

    def place_order(self, ticker: str, order_type: OrderType, amount: float) -> OrderResult:
        """Enhanced order placement with safety checks and logging."""
        if not self.config.should_trade_now(datetime.now(timezone.utc)):
            return OrderResult(success=False, error_message="Outside trading hours", amount=amount)

        try:
            if order_type == OrderType.BUY_RECOMMENDATION:
                if not self.has_sufficient_funds_available(amount):
                    return OrderResult(success=False, error_message="Insufficient funds", amount=amount)

                result = robinhood.orders.order_buy_fractional_by_price(
                    ticker, amount, timeInForce="gfd", extendedHours=False, jsonify=True
                )
            elif order_type == OrderType.SELL_RECOMMENDATION:
                if not self.has_sufficient_equity(ticker, amount):
                    return OrderResult(success=False, error_message="Insufficient equity", amount=amount)

                result = robinhood.orders.order_sell_fractional_by_price(
                    ticker, amount, timeInForce="gfd", extendedHours=False, jsonify=True
                )
            else:
                return OrderResult(success=False, error_message="Invalid order type", amount=amount)

            order_result = OrderResult(
                success=True,
                order_id=result.get("id"),
                details=result,
                amount=amount,
                price=float(result.get("price", 0.0)),
            )

            self.trade_history.append(order_result)
            self.last_trade_time[ticker] = datetime.now(timezone.utc)

            return order_result

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Order placement failed for %s: %s", ticker, e)
            return OrderResult(success=False, error_message=str(e), amount=amount)

    def execute_trade_decision(self, ticker: str) -> OrderResult:
        """Execute trade based on strategy signals and risk management."""
        signals = self.calculate_strategy_signals(ticker)
        final_signal = self.config.get_strategy_decision(signals)

        current_price = self.get_current_market_price(ticker)
        position = self.get_current_positions().get(ticker)

        # Check risk management if we have a position
        if position and self.check_risk_management(current_price, position):
            return self.place_order(ticker, OrderType.SELL_RECOMMENDATION, float(position["equity"]))

        # Calculate position size for new trades
        account_value = self.get_current_cash_position()
        volatility = self._calculate_volatility(ticker)
        position_size = self.config.get_position_size(account_value, volatility)

        if final_signal > self.config.threshold_percentage:
            return self.place_order(ticker, OrderType.BUY_RECOMMENDATION, position_size)
        elif final_signal < -self.config.threshold_percentage:
            if position:
                return self.place_order(ticker, OrderType.SELL_RECOMMENDATION, float(position["equity"]))

        return OrderResult(success=True, error_message="No trade conditions met", amount=0.0)

    def _calculate_volatility(self, ticker: str) -> float:
        """Calculate historical volatility for position sizing."""
        df = self.get_stock_history_dataframe(ticker, interval="day", span="month")
        if df.empty:
            return 0.0

        df["returns"] = df["close_price"].astype(float).pct_change()
        return df["returns"].std() * np.sqrt(252)

    def __enter__(self):
        """Context manager support for safe resource handling."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper logout on context exit."""
        self.robinhood_logout()
        if exc_type:
            logger.error("Error during execution: %s", exc_val)
            return False

    def validate_config(self) -> None:
        """Validate the trading configuration."""
        if not self.config.enabled_strategies:
            raise ValueError("No trading strategies enabled in the configuration.")

    def robinhood_logout(self) -> None:
        """Logout from Robinhood to ensure session is closed."""
        try:
            robinhood.logout()
            logger.info("Successfully logged out from Robinhood")
        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Failed to logout from Robinhood: %s", e)

    def get_stock_history_dataframe(self, ticker: str, interval: str, span: str) -> pd.DataFrame:
        """Fetch historical stock data from Robinhood and return as a DataFrame."""
        try:
            historicals = robinhood.stocks.get_stock_historicals(ticker, interval=interval, span=span, bounds="regular")

            if not historicals:
                logger.warning("No historical data available for %s", ticker)
                return pd.DataFrame()

            df = pd.DataFrame(historicals)

            # Convert string timestamps to datetime
            df["begins_at"] = pd.to_datetime(df["begins_at"])

            # Convert price columns to float
            price_columns = ["open_price", "close_price", "high_price", "low_price"]
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert volume to numeric
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            return df

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error fetching historical data: %s", str(e))
            return pd.DataFrame()

    def get_current_cash_position(self) -> float:
        """Retrieve the current cash position available for trading."""
        try:
            account = robinhood.account.build_user_profile()
            buying_power = float(account.get("buying_power", 0.0))
            cash = float(account.get("cash", 0.0))

            # Debug logging
            logger.debug("API Response - Account Data: %s", account)
            logger.debug("API Response - Buying Power: $%.2f, Cash: $%.2f", buying_power, cash)

            # Use the smaller of buying power or cash to be conservative
            available_cash = min(buying_power, cash)

            logger.info("Current available cash position: $%.2f", available_cash)
            return available_cash

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error fetching cash position: %s", str(e))
            return 0.0

    def check_risk_management(self, current_price: float, position: Dict) -> bool:
        """Check if the current position meets risk management criteria."""
        try:
            entry_price = float(position.get("average_buy_price", 0.0))
            quantity = float(position.get("quantity", 0.0))
            equity = current_price * quantity

            # Calculate position metrics
            unrealized_pl_pct = ((current_price - entry_price) / entry_price) * 100

            # Check stop loss
            if unrealized_pl_pct <= -self.config.risk_management.stop_loss_percentage:
                logger.info("Stop loss triggered at %.2f%% loss", unrealized_pl_pct)
                return True

            # Check trailing stop if enabled
            if self.config.risk_management.use_trailing_stop:
                highest_price = float(position.get("highest_price", entry_price))
                trailing_stop_pct = self.config.risk_management.trailing_stop_percentage

                if current_price <= highest_price * (1 - trailing_stop_pct / 100):
                    logger.info("Trailing stop triggered at %.2f", current_price)
                    return True

            # Check maximum position size
            if equity > self.config.risk_management.max_position_size:
                logger.info("Maximum position size exceeded: $%.2f", equity)
                return True

            return False

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error in risk management check: %s", str(e))
            return False

    def get_current_market_price(self, ticker: str) -> float:
        """Retrieve the current market price with SMA context."""
        try:
            current_price = self.get_current_market_price(ticker)

            if current_price <= 0:
                return 0.0

            # Get recent historical data for SMA context
            df = self.get_stock_history_dataframe(ticker, interval="5minute", span="day")

            if df.empty:
                return current_price

            # Calculate SMAs for validation
            short_term = self._calculate_sma_signal(df)
            long_term = self._calculate_sma_signal(df)

            # Only perform SMA validation if we have valid SMA values
            if short_term > 0:
                if abs(current_price - short_term) / short_term > 0.1:
                    logger.warning(
                        "Current price %s deviates significantly from short-term SMA %s",
                        current_price,
                        short_term,
                    )

            return current_price

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error retrieving current market price: %s", str(e))
            return self.get_current_market_price(ticker)

    def get_current_positions(self) -> Dict[str, Dict]:
        """Retrieve the current positions held in the account."""
        try:
            positions = robinhood.account.get_open_stock_positions()
            positions_dict = {}

            for position in positions:
                instrument_data = robinhood.stocks.get_instrument_by_url(position.get("instrument"))
                if not instrument_data:
                    continue

                ticker = instrument_data.get("symbol")
                if not ticker:
                    continue

                quantity = float(position.get("quantity", 0.0))
                average_buy_price = float(position.get("average_buy_price", 0.0))

                # Skip positions with zero quantity
                if quantity <= 0:
                    continue

                current_price = self.get_current_market_price(ticker)
                equity = quantity * current_price

                positions_dict[ticker] = {
                    "quantity": quantity,
                    "average_buy_price": average_buy_price,
                    "current_price": current_price,
                    "equity": equity,
                    "unrealized_pl": (current_price - average_buy_price) * quantity,
                    "unrealized_pl_pct": ((current_price - average_buy_price) / average_buy_price) * 100,
                    "highest_price": self._get_position_highest_price(ticker, position),
                }

            return positions_dict

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error fetching current positions: %s", str(e))
            return {}

    def _get_position_highest_price(self, ticker: str, position: Dict) -> float:
        """Helper method to get the highest price since position entry."""
        try:
            # Get historical data since position entry
            df = self.get_stock_history_dataframe(ticker, interval="day", span="year")
            if df.empty:
                return float(position.get("average_buy_price", 0.0))

            return df["high_price"].max()

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error calculating highest price: %s", str(e))
            return float(position.get("average_buy_price", 0.0))

    def has_sufficient_equity(self, ticker: str, amount: float) -> bool:
        """Check if there is sufficient equity for the given ticker and amount."""
        try:
            positions = self.get_current_positions()
            position = positions.get(ticker, {})

            if not position:
                return False

            current_equity = position.get("equity", 0.0)
            return current_equity >= amount

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error checking equity: %s", str(e))
            return False

    def has_sufficient_funds_available(self, amount: float) -> bool:
        """Check if there are sufficient funds available for the given amount."""
        try:
            available_cash = self.get_current_cash_position()

            # Add a small buffer (0.5%) to account for price fluctuations
            required_amount = amount * 1.005

            return available_cash >= required_amount

        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            logger.error("Error checking available funds: %s", str(e))
            return False
