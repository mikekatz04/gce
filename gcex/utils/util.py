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
from scipy import constants as ct

Msun = 1.989e30
rsun = 6.957 * (10 ** 5)


def gr_pdot(m1, m2, porb):
    Porb = porb * 3600.0 * 24.0  # days to seconds
    m1 = m1 * Msun
    m2 = m2 * Msun

    M_c = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    P_dot_gw = (
        -(96.0 * np.pi / (5 * ct.c ** 5))
        * 2 ** (8 / 3)
        * (ct.G * np.pi * M_c / Porb) ** (5 / 3)
    )

    return P_dot_gw
