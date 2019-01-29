from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from action_proposals.trainer import Trainer
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset

from action_proposals.models.bsn import Tem, TemLoss


class BSNTrainer(Trainer):

    def _get_optimizer(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        return optimizer

    def __init__(self):
        super(BSNTrainer, self).__init__()
        self.model = Tem(self.cfg.input_features)
        self.model = self.model.cuda()
        self.loss = TemLoss()

    def _add_user_config(self):
        self._parser.add_argument("--input_features", type=int, default=400)
        self._parser.add_argument("--csv_path", type=str, default="")
        self._parser.add_argument("--json_path", type=str, default="")
        self._parser.add_argument("--class_name_path", type=str, default="")
        self._parser.add_argument("--video_info_new_csv_path", type=str, default="")
        self._parser.add_argument("--learning_rate", type=float, default=1e-3)
        self._parser.add_argument("--batch_size", type=int, default=16)
        self._parser.add_argument("--weight_decay", type=float, default=1e-3)
        self._parser.add_argument("--num_workers", type=int, default=16)


    def _get_dataloader(self) -> DataLoader:
        self.dataset = ActivityNetDataset.get_ltw_feature_dataset(self.cfg.csv_path, self.cfg.json_path,
                                                                  self.cfg.video_info_new_csv_path,
                                                                  self.cfg.class_name_path, 'training')

        data_loader = DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        return data_loader

    def _train_one_epoch(self, data_loader: DataLoader, epoch: int, optimizer: Optimizer):

        for idx, (batch_feature, batch_proposals) in enumerate(data_loader):
            batch_feature = batch_feature.cuda()
            batch_proposals = batch_proposals.cuda()
            self.optimizer.zero_grad()
            batch_pred = self.model(batch_feature)
            loss_start, loss_action, loss_end = self.loss(batch_pred, batch_proposals)
            loss = (loss_start + 2 * loss_action + loss_end).mean()
            loss.backward()
            optimizer.step(None)

            # if idx % 100 == 0:
            #     print("epoch {}, iter {}: loss {}".format(epoch, idx, loss))

        print("epoch {}: loss start {}, action {}, end {}".format(epoch, loss_start, loss_action, loss_end))


if __name__ == '__main__':
    bsn_trainer = BSNTrainer()
    bsn_trainer.train()
