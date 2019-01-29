import unittest
import time
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset, BSNVideoRecordHandler
from torch.utils.data import DataLoader

class TestActivitynetDataset(unittest.TestCase):

    def test_activitynet_dataset(self):
        ltw_activitynet_dataset = ActivityNetDataset.get_ltw_feature_dataset(
            json_path='/home/dechao_meng/datasets/activitynet/annotations/anet_anno_action.json',
            csv_path='/home/dechao_meng/datasets/activitynet/csv_mean_100',
            class_name_path= '/home/dechao_meng/datasets/activitynet/annotations/action_name.csv'
        )
        assert (ltw_activitynet_dataset[0][0].shape == (400, 100))
        assert (len(ltw_activitynet_dataset) == 19228)

        # Retrival all the data use 81.54572629928589 seconds, about 30ms per csv.
        now = time.time()
        data_loader = DataLoader(ltw_activitynet_dataset, batch_size=128, shuffle=False, num_workers=16)
        for batch_data in data_loader:
            pass
        print('Retrival all the data use {} seconds'.format(time.time() - now))
        # Retrival all the data use 35.423325300216675 seconds
        now = time.time()
        for batch_data in data_loader:
            pass
        print('Retrival all the data use {} seconds'.format(time.time() - now))

