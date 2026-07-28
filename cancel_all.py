import ccxt
import time

def cancel_all_open_orders():
    print("[SYSTEM] Attempting to clean up ghost orders on Binance Testnet...")
    api_key = "CYDdL2sD4wsBy1g1mte1OieivnbBpuxwN63s0RoyYtxRLHjffGabECjvXmBcYacW"
    api_secret = "MSQRV7BnrVv28bJ6DkxtckXSpu8jkqZ38XuG8ASUjoueoMaAKJ7y31OqhggTV6NG"

    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True
        }
    })
    
    exchange.set_sandbox_mode(True)

    try:
        symbol = 'ETH/USDT'
        orders = exchange.fetch_open_orders(symbol)
        if not orders:
            print(f"[CLEANUP] No open orders found for {symbol}.")
            return
            
        print(f"[CLEANUP] Found {len(orders)} open ghost orders. Canceling...")
        
        # In ccxt, cancel_all_orders is universally supported by Binance
        try:
            exchange.cancel_all_orders(symbol)
            print("[SUCCESS] All open orders canceled.")
        except Exception as e:
            print(f"Failed bulk cancel. Falling back to individual cancel: {e}")
            for order in orders:
                exchange.cancel_order(order['id'], symbol)
                print(f" -> Canceled order {order['id']}")
                
    except Exception as e:
        print(f"[ERROR] Failed to clean up orders: {e}")

if __name__ == "__main__":
    cancel_all_open_orders()
