from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from action_proposals.trainer import Trainer
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset

from action_proposals.models.bsn import Tem, TemLoss


class BSNTrainer(Trainer):

    def _get_optimizer(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=0.01)
        return optimizer

    def __init__(self):
        super(BSNTrainer, self).__init__()
        self.model = Tem(self.cfg.input_features)
        self.loss = TemLoss()

    def _add_user_config(self):
        self._parser.add_argument("--input_features", type=int, default=400)
        self._parser.add_argument("--csv_path", type=str, default="")
        self._parser.add_argument("--json_path", type=str, default="")
        self._parser.add_argument("--class_name_path", type=str, default="")
        self._parser.add_argument("--learning_rate", type=float, default=1e-3)

    def _get_dataloader(self) -> DataLoader:
        self.dataset = ActivityNetDataset.get_ltw_feature_dataset(self.cfg.csv_path, self.cfg.json_path, self.cfg.class_name_path)
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def _train_one_epoch(self, data_loader: DataLoader, epoch: int, optimizer: Optimizer):

        for idx, (batch_feature, batch_proposals) in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss = (loss_start + 2 * loss_action + loss_end).sum()
            loss.backward()
            optimizer.step(None)

        print("epoch {}: loss {}, ".format(epoch, loss))


if __name__ == '__main__':
    bsn_trainer = BSNTrainer()
    bsn_trainer.train()