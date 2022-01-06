import time

import torch
from torch import as_tensor


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


class Fn:
    @staticmethod
    def train(model, data_loader, loss_fn, optimizer, device):
        start_time = time.time()
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

        return total_loss, hms_string(time.time() - start_time)

    @staticmethod
    def validation(model, data_loader, device):
        start_time = time.time()
        y_pred, y_true = [], []
        model.eval()
        with torch.no_grad():
            for index, (features, target) in enumerate(data_loader):
                y_pred.extend(model(features.to(device)))
                y_true.extend(target.to(device))

        return as_tensor(y_pred), as_tensor(y_true), hms_string(time.time() - start_time)
