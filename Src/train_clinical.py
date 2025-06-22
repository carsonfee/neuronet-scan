import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.train_torch import ClinicalNet
import numpy as np

# Mock data for illustration – replace with actual preprocessing pipeline
def load_mock_data():
    X = np.random.rand(100, 3).astype(np.float32)  # 100 samples, 3 features each
    y = np.random.randint(0, 2, size=(100,))
    return X, y

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
    X, y = load_mock_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wrap in PyTorch Datasets
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ClinicalNet(input_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                preds = model(val_x).argmax(dim=1)
                correct += (preds == val_y).sum().item()
                total += val_y.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Validation Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
  
