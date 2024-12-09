#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from src.bots.simple_moving_average import TradeBotSimpleMovingAverage
from src.bots.config import TradingConfig, RiskManagement
import time
import sys
from datetime import datetime, time as dt_time
import logging
import json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)


def load_tickers(filename="tickers.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "TECH": ["AAPL", "MSFT", "GOOGL"],
            "FINANCE": ["JPM", "BAC", "GS"],
            "HEALTHCARE": ["JNJ", "UNH", "PFE"],
            "ENERGY": ["XOM", "CVX", "COP"],
            "CONSUMER": ["PG", "KO", "WMT"],
        }


def is_market_open():
    now = datetime.now()
    current_time = dt_time(now.hour, now.minute)
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)

    # Add detailed logging
    is_weekday = now.weekday() < 5
    is_market_hours = market_open <= current_time <= market_close

    logging.info("Current time: %s", current_time)
    logging.info("Market hours: %s - %s", market_open, market_close)
    logging.info("Is weekday: %s", is_weekday)
    logging.info("Is during market hours: %s", is_market_hours)

    return is_weekday and is_market_hours


def main():
    config = TradingConfig(
        risk_management=RiskManagement(
            max_trade_amount=20.00,
        ),
        trade_interval_hours=4,
    )

    try:
        tb = TradeBotSimpleMovingAverage(config)
        sectors = load_tickers()

        logging.info("Starting enhanced trading bot...")
        logging.info("Initial cash position: $%.2f", tb.get_current_cash_position())

        while True:
            try:
                if not is_market_open():
                    logging.info("Market is closed. Waiting for market hours...")
                    time.sleep(300)
                    continue

                logging.info("Market is open. Starting trading operations...")

                portfolio_value = sum(float(pos["equity"]) for pos in tb.get_current_positions().values())

                for sector, tickers in sectors.items():
                    logging.info("\nAnalyzing %s sector...", sector)

                    for ticker in tickers:
                        try:
                            current_price = tb.get_current_market_price(ticker)
                            logging.info("Current price for %s: $%.2f", ticker, current_price)

                            if current_price > 0:
                                recommendation = tb.make_order_recommendation(ticker)

                                position = tb.get_current_positions().get(ticker, {})
                                position_value = float(position.get("equity", 0))

                                # Calculate position size based on portfolio
                                max_position = portfolio_value * config.risk_management.max_portfolio_per_stock
                                available_to_buy = max(0, max_position - position_value)
                                trade_amount = min(config.risk_management.max_trade_amount, available_to_buy)

                                logging.info("%s: $%.2f - %s", ticker, current_price, recommendation)

                                if recommendation.value == 1 and trade_amount >= 0:
                                    tb.execute_buy_order(ticker, trade_amount)
                                elif recommendation.value == 0 and position_value > 0:
                                    tb.execute_sell_order(
                                        ticker, min(config.risk_management.max_trade_amount, position_value)
                                    )
                                time.sleep(2)
                            else:
                                logging.warning("Unable to get valid price for %s", ticker)

                        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
                            logging.error("Error analyzing %s: %s", ticker, e)
                            continue

                logging.info("\nWaiting for next analysis cycle...")
                time.sleep(config.trade_interval_hours * 3600)

            except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
                logging.error("Error during trading loop: %s", e)
                time.sleep(300)

    except KeyboardInterrupt:
        logging.info("Gracefully shutting down...")
        tb.robinhood_logout()
        sys.exit(0)
    except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
