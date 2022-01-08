import torch
from torch import as_tensor


class Fn:
    @staticmethod
    def train(model, data_loader, loss_fn, optimizer, device):
        model.train()
        total_loss = 0

        for index, (features, target) in enumerate(data_loader):
            features, target = features.to(device), target.to(device)
            y = model(features)

            model.zero_grad()
            loss = loss_fn(y, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss

    @staticmethod
    def validation(model, data_loader, device):
        y_pred, y_true = [], []
        model.eval()
        with torch.no_grad():
            for index, (features, target) in enumerate(data_loader):
                y_pred.extend(model(features.to(device)))
                y_true.extend(target.to(device))

        return as_tensor(y_pred), as_tensor(y_true)

    @staticmethod
    def validation_score(model, data_loader, device, score_fn):
        y_pred, y_true = Fn.validation(model, data_loader, device)
        return score_fn(y_true.cpu().numpy(), y_pred.cpu().numpy())
