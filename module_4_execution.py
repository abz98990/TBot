# module_5_execution.py
import time


class ExecutionManager:
    def __init__(self, exchange_client):
        """
        Initializes the Execution Router with the Binance client.
        """
        self.exchange = exchange_client
        self.consecutive_failures = 0  # The 3-Strike Kill Switch Tracker
        self.max_failures = 3

    def calculate_targets(self, entry_price: float, side: str):
        """
        Calculates rigid Risk Management targets based on handwritten specs:
        - 5% Stop Loss (Damage Control)
        - TP1 (20%), TP2 (25%), TP3 (30%)
        """
        if side == 'BUY':
            sl = entry_price * 0.95  # 5% below entry
            tp1 = entry_price * 1.20  # 20% above entry
            tp2 = entry_price * 1.25  # 25% above entry
            tp3 = entry_price * 1.30  # 30% above entry
        else:  # SELL / SHORT
            sl = entry_price * 1.05
            tp1 = entry_price * 0.80
            tp2 = entry_price * 0.75
            tp3 = entry_price * 0.70

        return sl, tp1, tp2, tp3

    def verify_circuit_breaker(self) -> bool:
        """Checks if the system has suffered 3 consecutive failures."""
        if self.consecutive_failures >= self.max_failures:
            print("\n[FATAL] CIRCUIT BREAKER TRIGGERED! 3 Consecutive Failures Detected.")
            print("[FATAL] Halting all trading activity to protect user capital.")
            return False
        return True

    def process_signal(self, symbol: str, signal: str, current_price: float, trade_qty: float = 0.001):
        """
        Takes the AI signal and attempts to execute a live trade.
        """
        if not self.verify_circuit_breaker():
            return

        print(f"\n[EXECUTION] Processing {signal} signal for {symbol} at ${current_price:.2f}...")

        # Calculate Risk Parameters
        sl, tp1, tp2, tp3 = self.calculate_targets(current_price, signal)

        print(f"  --> [RISK CALC] Stop Loss (5%): ${sl:.2f}")
        print(f"  --> [RISK CALC] TP1 (20%): ${tp1:.2f} | TP2 (25%): ${tp2:.2f} | TP3 (30%): ${tp3:.2f}")

        # Live Order Routing (Wrapped in a try-except block to catch network/API errors)
        try:
            print(f"[EXECUTION] Dispatching Market {signal} Order to Binance Testnet...")

            # Note: For safety in this test, we use a hardcoded micro-quantity (e.g., 0.001 BTC)
            # In a production environment, this is calculated dynamically based on account balance.
            order_side = 'buy' if signal == 'BUY' else 'sell'

            # --- THE PHYSICAL TRADE ---
            main_order = self.exchange.create_market_order(symbol, order_side, trade_qty)
            filled_price = main_order.get('average') or current_price
            
            print(f"[SUCCESS] Entry Order Filled! Trade ID: {main_order['id']}")
            print(f"[SUCCESS] Executed at ${filled_price:.4f}")
            
            # --- RISK MANAGEMENT ORDERS (CCXT) ---
            # We place the opposites of the entry
            exit_side = 'sell' if signal == 'BUY' else 'buy'
            
            # Recalculate exact targets based on real filled price to be safe
            sl, tp1, tp2, tp3 = self.calculate_targets(filled_price, signal)
            
            print("[EXECUTION] Dispatching Stop-Loss & Take-Profit layers...")
            
            try:
                # Place Stop Loss Limit Order (Standard for Binance Spot)
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_LOSS_LIMIT',
                    side=exit_side,
                    amount=trade_qty,
                    price=sl,
                    params={'stopPrice': sl, 'timeInForce': 'GTC'}
                )
                print(f"  --> [SECURED] Stop-Loss Limit placed at ${sl:.2f}")
            except Exception as e_sl:
                print(f"[WARNING] Could not place Stop-Loss order: {e_sl}")
                
            try:
                # Place Limit Take Profit
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type='LIMIT',
                    side=exit_side,
                    amount=trade_qty,
                    price=tp1
                )
                print(f"  --> [SECURED] Take-Profit Limit placed at ${tp1:.2f}")
            except Exception as e_tp:
                print(f"[WARNING] Could not place Take-Profit order: {e_tp}")

            # Reset failures on a successful execution
            self.consecutive_failures = 0

        except Exception as e:
            self.consecutive_failures += 1
            print(f"[ERROR] Trade Execution Failed: {e}")
            print(f"[SYSTEM] Consecutive Failures: {self.consecutive_failures}/{self.max_failures}")