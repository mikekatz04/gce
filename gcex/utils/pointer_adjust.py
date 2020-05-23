#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Michael Katz (2020)
#
# This file is part of gce
#
# gce is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gce is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gce.  If not, see <http://www.gnu.org/licenses/>

import numpy as np

try:
    import cupy as cp

    gpu = True

except ImportError:
    gpu = False


def pointer_adjust(func):
    def func_wrapper(*args, **kwargs):
        targs = []
        for arg in args:
            if gpu:
                if isinstance(arg, cp.ndarray):
                    targs.append(arg.data.mem.ptr)
                    continue

            if isinstance(arg, np.ndarray):
                targs.append(arg.__array_interface__["data"][0])
                continue

            try:
                targs.append(arg.ptr)
                continue
            except AttributeError:
                targs.append(arg)
        # print(targs)
        return func(*targs, **kwargs)

    return func_wrapper
