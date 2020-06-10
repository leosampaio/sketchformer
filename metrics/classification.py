from sklearn.metrics import accuracy_score

from core.metrics import HistoryMetric


class PrecomputedValidationAccuracy(HistoryMetric):
    name = 'val-clas-acc'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, _ = input_data
        pred_labels = pred_y

        return accuracy_score(y, pred_labels)


class PrecomputedTestAccuracy(HistoryMetric):
    name = 'test-clas-acc'
    input_type = 'predictions_on_test_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, _ = input_data
        pred_labels = pred_y

        return accuracy_score(y, pred_labels)
