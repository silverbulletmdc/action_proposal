import unittest
import time
from action_proposals.dataset.activitynet_dataset import ActivityNetDataset, BSNVideoRecordHandler


class TestActivitynetDataset(unittest.TestCase):

    def test_activitynet_dataset(self):
        ltw_activitynet_dataset = ActivityNetDataset.get_ltw_feature_dataset(
            json_path='/home/dechao_meng/datasets/activitynet/annotations/anet_anno_action.json',
            csv_path='/home/dechao_meng/datasets/activitynet/csv_mean_100',
            class_name_path= '/home/dechao_meng/datasets/activitynet/annotations/action_name.csv'
        )
        assert (ltw_activitynet_dataset[0][0].shape == (100, 400))
        assert (len(ltw_activitynet_dataset) == 19228)

        # Retrival all the data use 727.5255792140961 seconds, about 30ms per csv.
        # now = time.time()
        # for i in range(len(ltw_activitynet_dataset)):
        #     data = ltw_activitynet_dataset[i]
        # print('Retrival all the data use {} seconds'.format(time.time() - now))

