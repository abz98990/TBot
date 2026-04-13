# module_3_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Defines the Neural Network Architecture.
        """
        super(LSTMPredictor, self).__init__()

        # The LSTM Layer: extracts temporal patterns from the sequence
        # batch_first=True ensures it expects our shape: (Samples, Time Steps, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Dropout prevents overfitting by randomly turning off neurons
        self.dropout = nn.Dropout(dropout)

        # The Dense Output Layer: compresses the LSTM's thoughts into 1 final prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Pass data through the LSTM
        out, _ = self.lstm(x)

        # We only care about the LSTM's output at the very LAST time step in the 60-candle window
        out = out[:, -1, :]

        # Pass through dropout and final linear layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


class ModelEngine:
    def __init__(self, input_size=3, learning_rate=0.001):
        """Initializes the training environment."""
        # Automatically use GPU if you have an Nvidia card, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODEL] Initializing Engine on device: {self.device}")

        self.model = LSTMPredictor(input_size=input_size).to(self.device)
        self.criterion = nn.MSELoss()  # Mean Squared Error
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_numpy, y_numpy, epochs=20, batch_size=32):
        """The training loop where the bot learns."""
        print(f"\n[MODEL] Commencing Training Phase ({epochs} Epochs)...")

        # 1. Convert NumPy arrays to PyTorch Tensors
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_numpy, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 2. Load into PyTorch Dataloader (Chronological order, DO NOT SHUFFLE time series)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 3. The Core Training Loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Reset gradients
                self.optimizer.zero_grad()

                # Make predictions
                predictions = self.model(batch_X)

                # Calculate how wrong the predictions were (Loss)
                loss = self.criterion(predictions, batch_y)

                # Backpropagation: Adjust the weights to be less wrong next time
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"   Epoch {epoch + 1:02d}/{epochs} | MSE Loss: {avg_loss:.6f}")

        print("[SUCCESS] Neural Network Training Complete.")

    def predict_next_candle(self, recent_window_X):
        """Generates a prediction for the absolute latest market data."""
        self.model.eval()  # Set to evaluation mode (turns off dropout for live inference)

        with torch.no_grad():  # Don't calculate gradients (saves memory/speed)
            # Convert the single numpy window into a PyTorch tensor
            # unsqueeze(0) adds a "batch" dimension, making it shape (1, 60, 3)
            X_tensor = torch.tensor(recent_window_X, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get the prediction
            predicted_log_return = self.model(X_tensor)

            # Convert back to a standard Python float
            return predicted_log_return.item()