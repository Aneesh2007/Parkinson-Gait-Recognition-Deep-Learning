import torch
import copy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
    patience=5,
):

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):

        # -----------------
        # Training
        # -----------------

        model.train()

        running_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # -----------------
        # Validation
        # -----------------

        model.eval()

        val_running_loss = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device).unsqueeze(1)

                outputs = model(X)

                loss = criterion(outputs, y)

                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # -----------------
        # Save Best Model
        # -----------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model = copy.deepcopy(model.state_dict())

            torch.save(best_model, "../models/mlp_best.pth")

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

             print(f"\nEarly stopping at epoch {epoch+1}")

             break
    model.load_state_dict(best_model)

    return model, history