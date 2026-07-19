from sklearn import metrics
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_model(model, test_loader, device):

    model.eval()

    predictions = []
    probabilities = []
    labels = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            outputs = model(X)

            probs = torch.sigmoid(outputs)

            preds = (probs >= 0.5).float()

            probabilities.extend(probs.cpu().numpy().flatten())
            predictions.extend(preds.cpu().numpy().flatten())
            labels.extend(y.numpy().flatten())

    print(f"Accuracy : {accuracy_score(labels, predictions):.4f}")
    print(f"Precision: {precision_score(labels, predictions):.4f}")
    print(f"Recall   : {recall_score(labels, predictions):.4f}")
    print(f"F1 Score : {f1_score(labels, predictions):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(labels, probabilities):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(labels, predictions))

    print("\nClassification Report")
    print(classification_report(labels, predictions))
    metrics = {
    "Accuracy": accuracy_score(labels, predictions),
    "Precision": precision_score(labels, predictions),
    "Recall": recall_score(labels, predictions),
    "F1": f1_score(labels, predictions),
    "ROC_AUC": roc_auc_score(labels, probabilities)
}

    return metrics