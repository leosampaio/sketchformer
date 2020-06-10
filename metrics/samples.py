import numpy as np

from core.metrics import ImageMetric
import utils


class ReconstructedSketchSamples(ImageMetric):
    name = 'sketch-reconstruction'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath, is_continuous = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(x))[:18]
        np.random.seed()

        x, pred_x = x[idx], pred_x[idx]
        if not is_continuous:
            x, pred_x = np.array(tokenizer.decode_list(x)), np.array(tokenizer.decode_list(pred_x))

        sketch_list = utils.sketch.build_interlaced_grid_list(
            x, pred_x, width=6)
        sketch_grid = utils.sketch.make_grid_svg(sketch_list)
        tmp_filepath = tmp_filepath.format('reconstruction')
        utils.sketch.draw_strokes(sketch_grid,
                                  svg_filename=plot_filepath.format('reconstruction'),
                                  png_filename=tmp_filepath)
        return tmp_filepath
