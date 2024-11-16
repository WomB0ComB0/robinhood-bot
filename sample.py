from src.bots.simple_moving_average import TradeBotSimpleMovingAverage
import time
import sys

def main():
    try:
        tb = TradeBotSimpleMovingAverage()
        trade_amount = 5.00  # Set a small fixed amount per trade
        ticker = "AAPL"     # You can modify this or add multiple tickers
        
        print("Starting trading bot...")
        print(f"Initial cash position: ${tb.get_current_cash_position()}")
        
        while True:
            try:
                current_positions = tb.get_current_positions()
                current_cash = tb.get_current_cash_position()
                current_price = tb.get_current_market_price(ticker)
                recommendation = tb.make_order_recommendation(ticker)
                
                print("\n" + "="*50)
                print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Current cash: ${current_cash:.2f}")
                print(f"Current positions: {current_positions}")
                print(f"{ticker} price: ${current_price:.2f}")
                print(f"Recommendation: {recommendation}")
                
                if recommendation.value == 1 and tb.has_sufficient_funds_available(trade_amount):
                    print(f"Placing buy order for ${trade_amount} of {ticker}")
                    tb.place_buy_order(ticker, trade_amount)
                elif recommendation.value == 0 and ticker in current_positions:
                    print(f"Placing sell order for ${trade_amount} of {ticker}")
                    tb.place_sell_order(ticker, trade_amount)
                
                # Wait for 1 hour before next check
                time.sleep(3600)  # 3600 seconds = 1 hour
                
            except Exception as e:
                print(f"Error during trading loop: {e}")
                time.sleep(300)  # Wait 5 minutes if there's an error
                
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        tb.robinhood_logout()
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
