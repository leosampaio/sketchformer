import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches


class PlotManager(object):

    def __init__(self):
        self.to_plot = {}

    def add_metric(self, name, data, data_type, skipped_iterations=None, legend=True,
                   x_label=None, y_label=None):
        self.to_plot[name] = {'data': data,
                              'data_type': data_type,
                              'skipped_iterations': skipped_iterations,
                              'has_legend': legend,
                              'x_label': x_label,
                              'y_label': y_label}

    def compute_grid_size(self):
        total_n_plots = len(self.to_plot)
        if total_n_plots == 1:
            grid_cols, grid_rows = 1, 1
        elif total_n_plots == 2:
            grid_cols, grid_rows = 2, 1
        elif total_n_plots == 3 or total_n_plots == 4:
            grid_cols, grid_rows = 2, 2
        elif total_n_plots == 5 or total_n_plots == 6:
            grid_cols, grid_rows = 2, 3
        elif total_n_plots == 7 or total_n_plots == 8 or total_n_plots == 9:
            grid_cols, grid_rows = 3, 3
        elif total_n_plots == 10 or total_n_plots == 11 or total_n_plots == 12:
            grid_cols, grid_rows = 4, 3
        elif total_n_plots == 13 or total_n_plots == 14 or total_n_plots == 15:
            grid_cols, grid_rows = 5, 3
        elif total_n_plots == 16:
            grid_cols, grid_rows = 4, 4
        elif total_n_plots == 17 or total_n_plots == 18 or total_n_plots == 19 or total_n_plots == 20:
            grid_cols, grid_rows = 5, 4
        elif total_n_plots == 21 or total_n_plots == 22 or total_n_plots == 23 or total_n_plots == 24:
            grid_cols, grid_rows = 6, 4
        return grid_cols, grid_rows

    def plot_lines(self, cell, metric_name, metric):
        ax = plt.subplot(cell)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        label = metric_name
        line, = ax.plot(list(
            range(0, len(metric['data']) * metric['skipped_iterations'],
                  metric['skipped_iterations'])),
            metric['data'], color='C0',
            label=label)
        lines = [line]
        if len(metric['data']) < 2:
            ax.text(0.5, 0.5, metric['data'][-1],
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=24)
        if metric['has_legend']:
            ax.legend(handles=lines, prop={'size': 16})

    def plot_scatter(self, cell, metric_name, metric):
        ax = plt.subplot(cell)
        ax.set_xticks([])
        ax.set_yticks([])

        cmap = cm.tab10
        category_labels = metric['data'][..., 2]
        unique_labels = list(set(category_labels))
        norm = colors.Normalize(vmin=0, vmax=len(unique_labels))
        for i, label in enumerate(unique_labels):
            category_labels = [i if x == label else x for x in category_labels]
            unique_labels = [i if x == label else x for x in unique_labels]

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        mapped_colors = cmapper.to_rgba(category_labels)

        ax.scatter(metric['data'][:, 0], metric['data'][:, 1],
                   color=mapped_colors,
                   label=unique_labels, alpha=0.7, marker='.',
                   edgecolors='none')
        patch = mpatches.Patch(color='silver', label=metric_name)
        ax.legend(handles=[patch], prop={'size': 20})

    def plot_image_grid(self, cell, metric_name, metric):
        imgs = metric['data']
        imgs = np.clip(imgs, 0., 1.)
        n_images = len(imgs)
        inner_grid_width = int(np.sqrt(n_images))
        inner_grid = GridSpecFromSubplotSpec(inner_grid_width, inner_grid_width, cell, wspace=0.1, hspace=0.1)
        is_first_one = True
        for i in range(n_images):
            inner_ax = plt.subplot(inner_grid[i])
            if is_first_one:
                inner_ax.set_title(metric_name)
                is_first_one = False
            if imgs.ndim == 4:
                inner_ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                inner_ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            inner_ax.axis('off')

    def plot_image(self, cell, metric_name, metric):
        ax = plt.subplot(cell)
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            image = plt.imread(metric['data'])
            ax.imshow(image)
        except OSError as e:
            print(repr(e))

    def plot_hist(self, cell, metric_name, metric):
        ax = plt.subplot(cell)
        ax.hist(metric['data'], bins=100)

    def plot(self, outfile, wspace=None, hspace=None, figsize=8):

        grid_cols, grid_rows = self.compute_grid_size()
        fig_w, fig_h = figsize * grid_cols, figsize * grid_rows

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(grid_rows, grid_cols)

        if wspace is not None:
            gs.update(wspace=wspace)
        elif hspace is not None:
            gs.update(hspace=hspace)

        for j, metric_name in enumerate(sorted(self.to_plot)):
            metric = self.to_plot[metric_name]
            current_cell = gs[j // grid_cols, j % grid_cols]

            if metric['data_type'] == 'lines':
                self.plot_lines(current_cell, metric_name, metric)
            elif metric['data_type'] == 'scatter':
                self.plot_scatter(current_cell, metric_name, metric)
            elif metric['data_type'] == 'image-grid':
                self.plot_image_grid(current_cell, metric_name, metric)
            elif metric['data_type'] == 'image':
                self.plot_image(current_cell, metric_name, metric)
            elif metric['data_type'] == 'hist':
                self.plot_hist(current_cell, metric_name, metric)

        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close(fig)
