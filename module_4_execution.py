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
        """
        if side == 'BUY':
            sl = entry_price * 0.95  # 5% below entry
        else:  # SELL / SHORT
            sl = entry_price * 1.05
        return sl

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
            return None

        print(f"\n[EXECUTION] Processing {signal} signal for {symbol} at ${current_price:.2f}...")

        # Calculate Risk Parameters
        sl = self.calculate_targets(current_price, signal)

        print(f"  --> [RISK CALC] Stop Loss (5%): ${sl:.2f}")

        # Live Order Routing
        try:
            print(f"[EXECUTION] Dispatching Market {signal} Order to Binance Testnet...")

            order_side = 'buy' if signal == 'BUY' else 'sell'

            # --- THE PHYSICAL TRADE ---
            main_order = self.exchange.create_market_order(symbol, order_side, trade_qty)
            filled_price = main_order.get('average') or current_price
            
            print(f"[SUCCESS] Entry Order Filled! Trade ID: {main_order['id']}")
            print(f"[SUCCESS] Executed at ${filled_price:.4f}")
            
            # --- RISK MANAGEMENT ORDERS ---
            exit_side = 'sell' if signal == 'BUY' else 'buy'
            sl = self.calculate_targets(filled_price, signal)
            
            print("[EXECUTION] Dispatching Stop-Loss layer...")
            sl_order_id = None
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
                sl_order_id = sl_order['id']
                print(f"  --> [SECURED] Stop-Loss Limit placed at ${sl:.2f}")
            except Exception as e_sl:
                print(f"[WARNING] Could not place Stop-Loss order: {e_sl}")

            # Reset failures on a successful execution
            self.consecutive_failures = 0

            return {
                'side': order_side,
                'amount': trade_qty,
                'entry_price': filled_price,
                'sl_id': sl_order_id
            }

        except Exception as e:
            self.consecutive_failures += 1
            print(f"[ERROR] Trade Execution Failed: {e}")
            print(f"[SYSTEM] Consecutive Failures: {self.consecutive_failures}/{self.max_failures}")
            return None

    def close_position(self, symbol: str, position: dict):
        """
        Forcefully closes an open position at market price and cleans up any open algo orders.
        """
        print(f"\n[EXECUTION] Closing {position['side'].upper()} position for {symbol} at market...")
        
        # 1. Cancel Stop Loss if it exists
        if position.get('sl_id'):
            try:
                self.exchange.cancel_order(position['sl_id'], symbol)
                print(f"  --> [CLEANUP] Cancelled Stop-Loss order {position['sl_id']}")
            except Exception as e:
                print(f"  --> [WARNING] Failed to cancel Stop-Loss (it may have already triggered): {e}")

        # 2. Market Close
        exit_side = 'sell' if position['side'] == 'buy' else 'buy'
        try:
            close_order = self.exchange.create_market_order(symbol, exit_side, position['amount'])
            filled_price = close_order.get('average') or 0.0
            if filled_price == 0.0:
                print("[WARNING] Could not get average fill price from exchange. Will use current market.")
            else:
                print(f"[SUCCESS] Position Closed! Exit Price: ${filled_price:.4f}")
            return filled_price
        except Exception as e:
            print(f"[ERROR] Failed to close position: {e}")
            return None