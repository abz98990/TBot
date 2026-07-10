import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

# Find available logs
csv_files = glob.glob("logs/*_performance*.csv")
if not csv_files:
    print("[ERROR] No performance logs found in 'logs/' directory. Run main.py first.")
    sys.exit(1)

# Default to the most recently modified CSV
target_csv = max(csv_files, key=os.path.getmtime)
if len(sys.argv) > 1:
    target_csv = sys.argv[1]

print(f"[SYSTEM] Monitoring Dashboard linked to: {target_csv}")
print("[SYSTEM] Close the window to stop tracking.")

# Set up the matplotlib figure
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.canvas.manager.set_window_title("T Bot - Real-Time Dashboard")
fig.suptitle(f"Live Performance Tracking: {os.path.basename(target_csv)}", fontsize=14, color='white')

# Create twin axis for MSE exactly ONCE outside the loop
ax3_mse = ax3.twinx()

def animate(i):
    try:
        if not os.path.exists(target_csv):
            return

        # Read the latest data
        df = pd.read_csv(target_csv)
        if df.empty:
            return

        # Keep only the last 60 data points to maintain readability (sliding window)
        df = df.tail(60)
        
        # We'll use the timestamp as X-axis labels if available
        x_labels = df['timestamp'].tolist() if 'timestamp' in df.columns else range(len(df))
        x = range(len(df))
        
        # 1. Top Panel: Asset Price
        ax1.clear()
        ax1.plot(x, df['last_price'], color='cyan', label='Asset Price', linewidth=1.5)
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle=':', alpha=0.3)
        
        # 2. Middle Panel: Prediction Confidence vs Actual Outcome
        ax2.clear()
        if 'actual_class' in df.columns and 'predicted_prob' in df.columns:
            ax2.plot(x, df['actual_class'], color='lime', label='Actual Outcome (1=UP, 0=DOWN)', linewidth=1.5, drawstyle='steps-post')
            ax2.plot(x, df['predicted_prob'], color='orange', label='LSTM Confidence (UP)', linewidth=1.5)
            ax2.axhline(0.5, color='white', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel("Probability")
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend(loc="upper left")
        elif 'actual_pct' in df.columns and 'predicted_pct' in df.columns:
            ax2.plot(x, df['actual_pct'], color='lime', label='Actual Realized %', linewidth=1.5)
            ax2.plot(x, df['predicted_pct'], color='orange', label='LSTM Predicted %', linestyle='--', linewidth=1.5)
            ax2.axhline(0, color='white', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel("Movement %")
            ax2.legend(loc="upper left")
        ax2.grid(True, linestyle=':', alpha=0.3)
        
        # 3. Bottom Panel: Cumulative Net PnL and Accuracy
        ax3.clear()
        ax3_mse.clear() # Clear the secondary axis as well
        
        if 'cumulative_pnl' in df.columns:
            ax3.plot(x, df['cumulative_pnl'], color='magenta', label='Cumulative Net PnL %', linewidth=2)
            ax3.axhline(0, color='white', linewidth=0.5, alpha=0.5)
            ax3.set_ylabel("Net PnL %")
            
        if 'accuracy' in df.columns:
            rolling_acc = df['accuracy'].rolling(window=5, min_periods=1).mean() * 100
            ax3_mse.plot(x, rolling_acc, color='lime', label='Rolling Accuracy (5-tick) %', linewidth=1.5, alpha=0.8)
            ax3_mse.set_ylabel("Accuracy %", color='lime')
            ax3_mse.set_ylim(-5, 105)
        elif 'mse' in df.columns:
            ax3_mse.plot(x, df['mse'], color='red', label='MSE (Deviation)', linewidth=1, alpha=0.5)
            ax3_mse.set_ylabel("MSE", color='red')
            
        # Combine legends safely if ax3 has lines
        lines_1, labels_1 = ax3.get_legend_handles_labels()
        lines_2, labels_2 = ax3_mse.get_legend_handles_labels()
        if lines_1 or lines_2:
            ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            
        ax3.set_xlabel("Time (last 60 ticks)")
        
        # Set x-ticks to show a few timestamps
        if len(x) > 0:
            tick_indices = [0, len(x)//2, len(x)-1]
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([x_labels[i] for i in tick_indices])
            
        ax3.grid(True, linestyle=':', alpha=0.3)
        
    except Exception as e:
        # Silently fail on read errors (e.g., if main script is currently writing to the file)
        pass

# Refresh every 2000 milliseconds (2 seconds)
ani = FuncAnimation(fig, animate, interval=2000, cache_frame_data=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
