import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from core.metrics import ProjectionMetric


class TSNEProjection(ProjectionMetric):
    name = 'tsne'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, _ = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        # [194, 103, 317, 100, 112, 221, 223, 293, 239, 8],
        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y, pred_z = y[idx], pred_z[idx]
        for label, feature in zip(y, pred_z):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                feats.append(feature)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)
        return np.concatenate((tsne_results, labels),
                              axis=1)


class PredictedLabelsTSNEProjection(ProjectionMetric):
    name = 'tsne-predicted'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, _ = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y, pred_z, pred_y = y[idx], pred_z[idx], pred_y[idx]
        for label, feature, pred_label in zip(y, pred_z, pred_y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                feats.append(feature)
                labels.append(pred_label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)
        return np.concatenate((tsne_results, np.expand_dims(labels, axis=1)),
                              axis=1)


class PCAProjection(ProjectionMetric):
    name = 'pca'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        entries, _, plot_filepath, tmp_filepath = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(entries))
        np.random.seed()

        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        for i in idx:
            skt = entries[i]
            if skt['label'] not in sel_labels and label_counter < 10:
                sel_labels.append(skt['label'])
                label_counter += 1
            if skt['label'] in sel_labels:
                feats.append(skt['features'])
                labels.append(skt['label'])
                counter += 1
            if counter >= 1000:
                break

        pca = PCA(n_components=2)
        pca.fit(feats)
        pca_result = pca.transform(feats)
        return np.concatenate((pca_result, np.expand_dims(labels, axis=1)),
                              axis=1)
