from sklearn import metrics
import torch
import numpy as np
from .pytorch_utils import forward


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader, max=50):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader, 
            return_target=True, max=max)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average='micro')
        auc = metrics.roc_auc_score(target, clipwise_output, average='micro')

        statistics = {'average_precision': average_precision,
                      'auc': auc}
        labels = (clipwise_output == clipwise_output.max(axis=-1)[:, None]).astype(int)

        try:
            micro_f1 = metrics.f1_score(target, labels, average='micro')
            statistics['micro_f1'] = micro_f1
        except Exception as e:
            print(e)
            statistics['micro_f1'] = 0
        return statistics, output_dict
