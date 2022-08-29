from cProfile import label
import torch
import matplotlib.pyplot as pl


class trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, test_loader, max_epochs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.losses_train = []
        self.losses_val = []
        self.device = "cpu"
        

    def fit(self):

        if torch.cuda.is_available():
            self.device = "cuda:0"
        
        self.model.to(self.device)
        for epoch in range(self.max_epochs):

            

            for i, batch in enumerate(self.train_loader, 0):

                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

            train_loss = self._evaluation(self.train_loader)
            val_loss = self._evaluation(self.val_loader)

            self.losses_train.append(train_loss)
            self.losses_val.append(val_loss)
            print(train_loss, " , ", val_loss)

        
    def _early_stopping(self, epoch, best_val, best_model_state, best_optimizer_state, val_loss, train_loss, stop):
        pass
    def _evaluation(self, data_loader):

        self.model.eval()
        losses = []
        
        for i, batch in enumerate(data_loader, 0):
            
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
            losses.append(loss.cpu().numpy())
        
        data_loss = losses.mean()
        self.model.train()

        return data_loss
    
    def plot(self):

        epochs = list(range(1, len(self.losses_train)+1))
        
        pl.figure(figsize=(5, 2.7), layout='constrained')
        pl.plot(epochs, self.losses_train, label='train_loss')
        pl.plot(epochs, self.losses_val, label='val_loss')
        pl.xlabel('epoch')
        pl.ylabel('loss')
        pl.title("Train/Val Loss")
        pl.legend()