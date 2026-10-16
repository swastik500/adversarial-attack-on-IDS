import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

from utils import create_datasets, IDS_Dataset
from models import IDS_model, IDS_CNN, IDS_RNN

def train_model(model_name, model, train_loader, test_loader, epochs, lr, device):
    print(f"\n--- Training {model_name} ---")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)
            
        epoch_loss = train_loss / total
        epoch_acc = (correct / total) * 100
        
        # Test evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * x_batch.size(0)
                preds = (outputs >= 0.5).float()
                test_correct += (preds == y_batch).sum().item()
                test_total += x_batch.size(0)
        
        val_loss = test_loss / test_total
        val_acc = (test_correct / test_total) * 100
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    os.makedirs("trained_models", exist_ok=True)
    save_path = f"trained_models/{model_name.lower()}_ids_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {model_name} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train MLP, CNN, RNN, and GBT classifiers for Intrusion Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--model", type=str, default="all", choices=["mlp", "cnn", "rnn", "gbt", "all"], help="Specific model to train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the create_datasets loader from utils
    x_train, y_train, x_test, y_test = create_datasets(train_ratio=0.8, b_m_ratio=2)
    print(f"Data split: Train={x_train.shape}, Test={x_test.shape}")

    train_dataset = IDS_Dataset(x_train, y_train)
    test_dataset = IDS_Dataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    target = args.model.lower()

    # 1. Train MLP Model
    if target in ("mlp", "all"):
        mlp_model = IDS_model()
        train_model("MLP", mlp_model, train_loader, test_loader, args.epochs, args.lr, device)

    # 2. Train CNN Model
    if target in ("cnn", "all"):
        cnn_model = IDS_CNN()
        train_model("CNN", cnn_model, train_loader, test_loader, args.epochs, args.lr, device)

    # 3. Train RNN Model
    if target in ("rnn", "all"):
        rnn_model = IDS_RNN()
        train_model("RNN", rnn_model, train_loader, test_loader, args.epochs, args.lr, device)

    # 4. Train GBT Model
    if target in ("gbt", "all"):
        from sklearn.ensemble import GradientBoostingClassifier
        import pickle
        print("\n--- Training GBT ---")
        gbt = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gbt.fit(x_train, y_train)
        val_acc = gbt.score(x_test, y_test) * 100
        print(f"GBT Validation Accuracy: {val_acc:.2f}%")
        os.makedirs("trained_models", exist_ok=True)
        save_path = "trained_models/gbt_ids_model.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(gbt, f)
        print(f"Saved GBT to {save_path}")

if __name__ == "__main__":
    main()
