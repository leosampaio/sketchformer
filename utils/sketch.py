# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SketchRNN data loading and image manipulation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
import os
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def get_bounds(data, factor=1):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def get_absolute_bounds(data, factor=1):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return min_x, max_x, min_y, max_y


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def scale_bound(stroke, average_dimension=10.0):
    """Scale an entire image to be less than a certain size."""
    # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
    # modifies stroke directly.
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_binary_stroke5(s):
    s = np.array(s, dtype=np.float32)
    one_hot = np.argmax(s[:, 2:], axis=-1)
    s[:, 2:] = 0
    s[range(one_hot.shape[0]), one_hot + 2] = 1.
    return s


def convert_to_absolute(sketch):
    absolute_sketch = np.zeros_like(sketch)
    absolute_sketch[0] = sketch[0]
    for i, (prev, new, orig) in enumerate(zip(absolute_sketch, absolute_sketch[1:], sketch[1:])):
        new[:2] = prev[:2] + orig[:2]
        new[2:] = orig[2:]
    return absolute_sketch


def to_relative(sketch, factor=1):
    relative_sketch = np.zeros_like(sketch)
    relative_sketch[0] = sketch[0]
    relative_sketch[0, :2] = sketch[0, :2] * factor
    for i, (prev_orig, new, orig) in enumerate(zip(sketch, relative_sketch[1:], sketch[1:])):
        new[:2] = (orig[:2] - prev_orig[:2]) * factor
        new[2:] = orig[2:]
    return relative_sketch


def list_to_relative(sketches):
    relative_sketches = []
    for s in sketches:
        relative_sketches.append(to_relative(s))
    return relative_sketches


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if np.argmax(big_stroke[i, :]) == 4:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def predictions_to_sketches(preds):
    return np.array([to_normal_strokes(to_binary_stroke5(p)) for p in preds])


def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
                int(round(sample_strokes[j][0] * factor)),
                int(round(sample_strokes[j][1] * factor)),
                int(sample_strokes[j][2]),
                int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


def draw_strokes(data, factor=0.01, svg_filename='/tmp/sketch_rnn/svg/sample.svg',
                 png_filename='/tmp/sketch_rnn/svg/sample.png'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 2.
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    drawing = svg2rlg(svg_filename)
    renderPM.drawToFile(drawing, png_filename, fmt="PNG")


def make_grid_svg(s_list, grid_space=2.25, grid_space_x=2.5):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end
    x_pos = 0.0
    y_pos = 0.0
    result = []
    for sample in s_list:
        sketch = sample[0]
        if len(sketch) == 0:
            continue
        sketch[0, -1] = 1
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(sketch)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += sketch.tolist()
        if result[-1][2] == 1:
            result[-2][2] = 0
        else:
            result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def build_interlaced_grid_list(targets, preds, width=8):
    grid_list = []
    current_sketch = 0
    for i in range(0, width, 2):
        for j in range(width):
            grid_list.append([targets[current_sketch], [i, j]])
            try:
                grid_list.append([preds[current_sketch], [i + 1, j]])
            except:
                pass
            current_sketch += 1
    return grid_list


def build_interlaced_grid_list_3_lines(a, b, c, width=9):
    grid_list = []
    current_sketch = 0
    for i in range(0, width, 3):
        for j in range(width):
            grid_list.append([a[current_sketch], [i, j]])
            try:
                grid_list.append([b[current_sketch], [i + 1, j]])
                grid_list.append([c[current_sketch], [i + 2, j]])
            except:
                pass
            current_sketch += 1
    return grid_list


def build_grid_list(sketches, width=8):
    grid_list = []
    current_sketch = 0
    for i in range(0, width):
        for j in range(width):
            grid_list.append([sketches[current_sketch], [i, j]])
            current_sketch += 1
    return grid_list


def composition_to_lines(sketches, sketch_sizes, sketch_positions, scale=1.0):
    """
    convert strokes3 to polyline format ie. absolute x-y coordinates
    note: the sketch can be negative
    :param strokes: stroke3, Nx3
    :param scale: scale factor applied on stroke3
    :param start_from_origin: sketch starts from [0,0] if True
    :return: list of strokes, each stroke has format Nx2
    """
    x = 0
    y = 0
    lines = []
    for sketch, size, pos in zip(sketches, sketch_sizes, sketch_positions):
        position = np.array(pos) * scale
        x, y = position
        line = [position]
        for i in range(len(sketch)):
            x_, y_ = sketch[i, :2] * scale * (size + .1)
            x += x_
            y += y_
            line.append([x, y])
            if sketch[i, 2] == 1:
                line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
                lines.append(line_array)
                line = []
        if line:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
    if lines == []:
        line_array = [np.zeros((1, 2), dtype=np.uint8)]
        lines.append(line_array)
    return lines
