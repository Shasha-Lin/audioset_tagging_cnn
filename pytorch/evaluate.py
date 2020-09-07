from sklearn import metrics
import numpy as np
from .pytorch_utils import forward


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
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
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        labels = output_dict['labels']
        statistics = {'average_precision': average_precision,
                      'auc': auc}

        try:
            micro_f1 = metrics.f1_score(target, labels, average='micro')
            statistics['micro_f1'] = micro_f1
        except:
            pass

        return statistics
