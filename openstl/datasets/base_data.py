import lightning as l


class BaseDataModule(l.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.test_mean = test_loader.dataset.mean
        self.test_std = test_loader.dataset.std
        self.data_name = getattr(test_loader.dataset, 'dataname', 'weather')

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader