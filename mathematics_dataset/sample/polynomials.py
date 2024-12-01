# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate polynomials with given values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

# Dependency imports
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.util import combinatorics
import numpy as np
import six
from six.moves import range
from six.moves import zip
import sympy
from sympy import symbols, diophantine


def expanded_coefficient_counts(length, is_zero):
    """Generates list of integers for number of terms of given power."""
    if length == 1 and all(is_zero):
        raise ValueError('length=1 and all zero')

    counts = np.asarray([0 if zero else 1 for zero in is_zero])
    extra_needed = (length - sum(counts))

    if extra_needed < 0:
        raise ValueError('length={} cannot handle is_zero={}'
                         .format(length, is_zero))

    extra = combinatorics.uniform_non_negative_integers_with_sum(
        count=len(is_zero), sum_=extra_needed)
    counts += np.asarray(extra)

    # Tweak so that no zeros get "1".
    while True:
        bad_zeros = [
            i for i in range(len(is_zero)) if is_zero[i] and counts[i] == 1
        ]
        if not bad_zeros:
            break
        take_from = random.choice(bad_zeros)
        add_to = random.choice(
            [i for i in range(len(is_zero)) if counts[i] >= 1 and i != take_from])
        counts[take_from] -= 1
        counts[add_to] += 1

    return counts


def diophantine_solve_linear_2d(c, a, b, t):
    """
    Solves a linear Diophantine equation: a*x + b*y = c.
    Returns a particular solution (x, y).
    """
    # Define variables
    x, y = symbols('x y', integer=True)

    # Solve the equation
    solutions = diophantine(a * x + b * y - c)
    if solutions:
        # Extract the first solution
        particular_solution = next(iter(solutions))
        x_sol, y_sol = particular_solution

        # Apply the shift (t) to x and y
        return x_sol + t * b, y_sol - t * a
    else:
        raise ValueError(f"No solution for equation {a}*x + {b}*y = {c}")


def coefficients_linear_split(coefficients, entropy):
    """Finds two sets of coefficients and multipliers summing to `coefficients`."""
    coefficients = np.asarray(coefficients)
    coefficients_shape = coefficients.shape
    coefficients = np.reshape(coefficients, [-1])

    entropy_a = max(1, random.uniform(0, entropy / 3))
    entropy_b = max(1, random.uniform(0, entropy / 3))
    entropy -= entropy_a + entropy_b
    entropy_coefficients = entropy * np.random.dirichlet(np.ones(len(coefficients)))

    # For each target coefficient z, we are required to solve the linear
    # Diophantine equation a*x + b*y = c. Bezout's theorem: this has a solution if
    # and only if gcd(a, b) divides c.
    coefficients_gcd = sympy.gcd([i for i in coefficients])
    coefficients_gcd = max(1, abs(coefficients_gcd))

    a = number.integer(entropy_a, signed=True, min_abs=1)
    b = number.integer(entropy_b, signed=True, min_abs=1, coprime_to=a)
    b *= _random_factor(coefficients_gcd)
    if random.choice([False, True]):
        a, b = b, a

    coefficients_1 = np.zeros(coefficients.shape, dtype=np.object)
    coefficients_2 = np.zeros(coefficients.shape, dtype=np.object)

    for index, coefficient in enumerate(coefficients):
        entropy_coeff = entropy_coefficients[index]
        t = number.integer(entropy_coeff, signed=True)
        try:
            x, y = diophantine_solve_linear_2d(c=coefficient, a=a, b=b, t=t)
            coefficients_1[index] = x
            coefficients_2[index] = y
        except ValueError as e:
            print(f"Warning: {e}")
            coefficients_1[index] = 0
            coefficients_2[index] = 0

    # Prevent all coefficients from being zero.
    while np.all(coefficients_1 == 0) or np.all(coefficients_2 == 0):
        index = random.randint(0, len(coefficients) - 1)
        scale = random.choice([-1, 1])
        coefficients_1[index] += scale * b
        coefficients_2[index] -= scale * a

    coefficients_1 = np.reshape(coefficients_1, coefficients_shape)
    coefficients_2 = np.reshape(coefficients_2, coefficients_shape)

    return a, b, coefficients_1, coefficients_2


def _random_factor(integer):
    factors = sympy.factorint(integer)
    result = 1
    for factor, power in six.iteritems(factors):
        result *= factor ** random.randint(0, power)
    return result
