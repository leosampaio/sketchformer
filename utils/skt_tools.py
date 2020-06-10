#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:25:54 2018

@author: Tu Bui tb00083@surrey.ac.uk
"""
import numpy as np
import os
import svgwrite
# import json
from PIL import Image, ImageDraw
from rdp import rdp # pip install rdp
# from IPython.display import SVG, display
from svgpathtools import real, imag, svg2paths, wsvg  # pip install git+https://github.com/mathandy/svgpathtools#egg=svgpathtools


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    if so < 1e-6:  # p0 = p1
        return p0
    else:
        return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


def get_bounds(data, factor=1.0):
    """Return bounds of stroke-3 data."""
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

    return min_x, max_x, min_y, max_y


def lines_to_strokes(lines, omit_first_point=True):
    """
    Convert polyline format to stroke-3 format.
    lines: list of strokes, each stroke has format Nx2
    """
    strokes = []
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :] if omit_first_point else strokes


def strokes_to_lines(strokes, scale=1.0, start_from_origin=False):
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
    line = [[0, 0]] if start_from_origin else []
    for i in range(len(strokes)):
        x_, y_ = strokes[i, :2] * scale
        x += x_
        y += y_
        line.append([x, y])
        if strokes[i, 2] == 1:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
            line = []
    if lines == []:
        line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
        lines.append(line_array)
    return lines


def centralise_lines(lines, shape=None, jitter=False):
    """
    put polyline in centre of a canvas specified by shape
    :param lines: list of strokes each having format Nx2 (e.g. output of strokes_to_lines)
    :param shape: shape of the canvas (x,y); None if you want the canvas auto fit the lines
    :param jitter: if True, random offset within shape canvas
    :return: lines after centred and offset
    """
    # find line boundary
    line_array = np.concatenate(lines, axis=0)  # Nx2
    min_x, min_y = line_array.min(axis=0)
    max_x, max_y = line_array.max(axis=0)
    if shape is None:
        shape = (max_x - min_x, max_y - min_y)
    if jitter:
        dx = max(int((shape[0] - max_x + min_x)/2.0) - 2, 0)
        dy = max(int((shape[1] - max_y + min_y)/2.0) - 2, 0)
        offset = (np.random.randint(-dx, dx+1), np.random.randint(-dy, dy+1))
    else:
        offset = (0, 0)
    sx = offset[0] + (shape[0] - max_x - min_x)/2  # total shift along x
    sy = offset[1] + (shape[1] - max_y - min_y)/2  # --- y
    sxy = np.array([sx, sy])[None, ...]
    out = [line + sxy for line in lines]
    return out


def normalise_strokes3(stroke3, max_bound=1.0):
    """
    normalise skt to max_bound
    :param stroke3: stroke3 format (N,3)
    :param max_bound: max len
    :return: (N, 3)
    """
    stroke = np.array(stroke3, dtype=np.float32)
    min_x, max_x, min_y, max_y = get_bounds(stroke)
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    stroke[:, :2] = stroke[:, :2] / max_dim * max_bound
    return stroke


def aggregate(strokes, max_bound=1.0):
    """
    concat stroke3 data into a single array
    also rescale to have max_bound
    used to create a hdf5 database

    :param strokes: array of stroke-3, length N
    :param max_bound: maximum bound of sketch along x and y dimension
    :return: (concated, ids, N) where
        concated: all data concatenated in a single array
        ids: has size Nx2 showing start and end position in concated
        N: number of datum
    """
    N = len(strokes)
    # get start and end position
    dlen = [len(x) for x in strokes]
    ids = np.repeat(np.cumsum(dlen), 2).tolist()
    ids = [0, ] + ids[:-1]
    ids = np.int64(ids).reshape((N, 2))
    # rescale
    strokes_norm = []
    for i in range(N):
        stroke = np.array(strokes[i], dtype=np.float32)
        min_x, max_x, min_y, max_y = get_bounds(stroke)
        max_dim = max([max_x - min_x, max_y - min_y, 1])
        stroke[:, :2] = stroke[:, :2] / max_dim * max_bound
        strokes_norm.append(stroke)
    # concatenate
    concated = np.concatenate(strokes_norm, axis=0)
    return concated, ids, N


def read_svg(svg_path, scale=100.0, draw_mode=False):
    """
    read svg, centralised and convert to stroke-3 format
    scale: stroke-3 output having max dimension [-scale, +scale]
    """
    try:
        paths, path_attrs = svg2paths(svg_path, return_svg_attributes=False)  # svg to paths
        lines = []
        lens = []
        for path_id, path in enumerate(paths):  # get poly lines from path
            erase = False  # path could be erased by setting stroke attribute to #fff (sketchy)
            path_attr = path_attrs[path_id]
            if 'stroke' in path_attr and path_attr['stroke'] == '#fff':
                erase = True
            # try:
            plen = int(path.length())
            # except ZeroDivisionError:
            #     plen = 0
            if plen > 0 and not erase:
                lines.append([path.point(i) for i in np.linspace(0, 1, max(2, plen))])
                lens.append(plen)

        # convert to (x,y) coordinates
        lines = [np.array([[real(x), imag(x)] for x in path]) for path in lines]

        # get dimension of this drawing
        tmp = np.concatenate(lines, axis=0)
        w_max, h_max = np.max(tmp, axis=0)
        w_min, h_min = np.min(tmp, axis=0)
        w = w_max - w_min
        h = h_max - h_min
        max_hw = max(w, h)

        def group(line):
            out = np.array(line, dtype=np.float32)
            out[:, 0] = ((out[:, 0] - w_min) / max_hw * 2.0 - 1.0) * scale
            out[:, 1] = ((out[:, 1] - h_min) / max_hw * 2.0 - 1.0) * scale
            return out

        # normalised
        lines = [group(path) for path in lines]
        lines_simplified = [rdp(path, epsilon=1.5) for path in lines]  # apply RDP algorithm

        strokes_simplified = lines_to_strokes(lines_simplified)  # convert to 3-stroke format (dx,dy,pen_state)
        # scale_bound(strokes_simplified, 10)
        if draw_mode:
            draw_strokes3(strokes_simplified, 1.0)  # no need to concat the origin point
            print('num points: {}'.format(len(strokes_simplified)))
        return np.array(strokes_simplified, dtype=np.float32)
    except Exception as e:
        print('Error encountered: {} - {}'.format(type(e), e))
        print('Location: {}'.format(svg_path))
        raise


def draw_strokes3(data, factor=0.2, svg_filename='test.svg', stroke_width=1):
    """
    draw stroke3 to svg
    :param data: stroke3, add origin (0,0) if doesn't have
    :param factor: scale factor
    :param svg_filename: output file
    :return: None
    """
    if np.abs(data[0]).sum() != 0:
        data2 = np.r_[np.zeros((1, 3), dtype=np.float32), data]
    else:
        data2 = data
    parent_dir = os.path.dirname(svg_filename)
    if parent_dir and not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    min_x, max_x, min_y, max_y = get_bounds(data2, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data2)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data2[i, 0]) / factor
        y = float(data2[i, 1]) / factor
        lift_pen = data2[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    # display(SVG(dwg.tostring()))


def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
    """draw a grid of svg given a list of sketches in stroke-3 format"""
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max+x_min)*0.5
        return x_start-center_loc, x_end
    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0]*grid_space+grid_space*0.5
        grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x+loc_x
        new_y_pos = grid_y+loc_y
        result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos+delta_pos[0]
        y_pos = new_y_pos+delta_pos[1]
    return np.array(result)


def draw_strokes_xy(stroke_list, image_shape=(256, 256), background_pixel=255,
                    colour=False, line_width=2):
    """
    draw image from raw ndjson/simplified/csv or strokes in (x,y) format
    :param stroke_list: (list) either 2xN or Nx2, must be scaled to fit image_shape
    :param image_shape: output shape.
    :param background_pixel: {255,0}
    :param colour: (bool) return 3 or 1 channel image
    :param line_width: (int) linewidth of strokes
    :return: image as numpy
    """
    strokes = np.copy(stroke_list)  # avoid modifying destructively
    # check if stroke is of size 2xN (column order) or Nx2 (row order)
    tmp = np.unique([len(stroke) for stroke in strokes])
    col_order = 1 if tmp.size == 1 else 0

    min_xy = np.min(np.concatenate(strokes, axis=col_order), axis=col_order)
    if np.any(min_xy < 0):
        strokes = [stroke - min_xy for stroke in strokes]
    im = Image.new('L', image_shape, background_pixel)
    draw = ImageDraw.Draw(im)
    fill = 255 - background_pixel
    if col_order:
        for stroke in strokes:
            draw.line(list(zip(stroke[0], stroke[1])), fill=fill, width=line_width)
    else:
        for stroke in strokes:
            draw.line(list(zip(stroke[:, 0], stroke[:, 1])), fill=fill, width=line_width)

    im = np.array(im)
    if colour:
        return np.repeat(im[..., None], 3, axis=2)
    return im


def draw_lines(lines, image_shape=(256, 256), background_pixel=255, colour=False,
               line_width=2, typing='int'):
    """
    a fast version of draw_strokes_xy() assuming xy order
    :param lines: list of strokes, each has format Nx2
    :param image_shape: output image shape
    :param background_pixel: {255,0}
    :param colour: (bool) return 3 or 1 channel image
    :param line_width: (int) linewidth of strokes
    :return: image as numpy
    """
    # strokes = [line + np.array(offset)[None, ...] for line in lines]
    image_mode = 'L' if typing == 'int' else 'F'
    im = Image.new(image_mode, image_shape, background_pixel)
    draw = ImageDraw.Draw(im)
    if typing == 'int':
        fill_value = 255 - background_pixel
    elif typing == 'float':
        fill_value = 1.0 - background_pixel
    for line in lines:
        draw.line(list(zip(line[:, 0], line[:, 1])), fill=fill_value, width=line_width)
    im = np.array(im)
    if colour:
        return np.repeat(im[..., None], 3, axis=2)
    return im


def svg_to_png(in_svg, out_png):
    from cairosvg import svg2png
    svg2png(open(in_svg, 'rb').read(), write_to=open(out_png, 'wb'))
