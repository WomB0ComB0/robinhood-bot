from src.bots.simple_moving_average import TradeBotSimpleMovingAverage
import time
import sys
from datetime import datetime, time as dt_time

def is_market_open():
    """Check if the market is currently open"""
    now = datetime.now()
    current_time = dt_time(now.hour, now.minute)
    market_open = dt_time(9, 30)    # 9:30 AM EST
    market_close = dt_time(16, 0)   # 4:00 PM EST
    
    # Check if it's a weekday and within market hours
    return now.weekday() < 5 and market_open <= current_time <= market_close

def main():
    try:
        tb = TradeBotSimpleMovingAverage()
        trade_amount = 5.00  # Set a small fixed amount per trade
        
        # Diversified portfolio across different sectors
        tickers = {
            # Technology
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            # Communication Services
            "GOOGL": "Alphabet",
            "META": "Meta Platforms",
            # Consumer Discretionary
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            # Healthcare
            "JNJ": "Johnson & Johnson",
            "UNH": "UnitedHealth",
            # Financial
            "BRK-B": "Berkshire Hathaway",
            "JPM": "JPMorgan Chase",
            # Industrial
            "CAT": "Caterpillar",
            "BA": "Boeing",
            # Consumer Staples
            "PG": "Procter & Gamble",
            "KO": "Coca-Cola",
            # Energy
            "XOM": "Exxon Mobil",
            "CVX": "Chevron"
        }
        
        print("Starting long-term trading bot...")
        print(f"Initial cash position: ${tb.get_current_cash_position():.2f}")
        print("Using 180-day and 365-day moving averages for analysis")
        
        while True:
            try:
                if not is_market_open():
                    print(f"\nMarket is closed. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("Waiting for next market day...")
                    time.sleep(3600)  # Check every hour
                    continue
                
                current_positions = tb.get_current_positions()
                current_cash = tb.get_current_cash_position()
                
                print("\n" + "="*70)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Current cash: ${current_cash:.2f}")
                print(f"Current positions: {current_positions}")
                
                for ticker, company in tickers.items():
                    try:
                        current_price = tb.get_current_market_price(ticker)
                        recommendation = tb.make_order_recommendation(ticker)
                        
                        print(f"\nAnalyzing {company} ({ticker}) at ${current_price:.2f}")
                        
                        if recommendation.value == 1 and tb.has_sufficient_funds_available(trade_amount):
                            print(f"🟢 Placing buy order for ${trade_amount} of {ticker}")
                            tb.place_buy_order(ticker, trade_amount)
                        elif recommendation.value == 0 and ticker in current_positions:
                            print(f"🔴 Placing sell order for ${trade_amount} of {ticker}")
                            tb.place_sell_order(ticker, trade_amount)
                        else:
                            print(f"⚪ Holding position in {ticker}")
                        
                        time.sleep(5)  # Small delay between stocks
                        
                    except Exception as e:
                        print(f"Error analyzing {ticker}: {e}")
                        continue
                
                print("\nCompleted analysis of all stocks")
                print("Waiting 4 hours before next analysis...")
                time.sleep(14400)  # Wait 4 hours before next round
                
            except Exception as e:
                print(f"Error during trading loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
                
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        tb.robinhood_logout()
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
