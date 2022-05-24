import math

cimport numpy as np
import numpy as np

cimport cython
from cpython cimport bool


def online_shapelet_distance(np.ndarray series, np.ndarray shapelet, list sorted_indicies, int position, int length):

    cdef:
        np.float64_t sum = 0.0
        np.float64_t sum2 = 0.0
        np.float64_t best_dist = 0
        np.float64_t dist = 0
        np.float64_t mean, std
        np.float64_t start, end
        np.float64_t temp, val
        int n, mod, j
        bool use_std

    subseq = series[position: position + length]

    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = (sum2 - mean * mean * length) / length
    if std > 0:
        subseq = (subseq - mean) / std

    else:
        subseq = np.zeros_t(length)

    # best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt((sums2[n] - mean * mean * length) / length)

            # dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


def calc_early_binary_ig(
    orderline,
    c1_traversed,
    c2_traversed,
    c1_to_add,
    c2_to_add,
    worst_quality,
):
    cdef:
        np.float64_t initial_ent, left_prop, right_prop, ent_left, ent_right, ig
        np.float64_t bsf_ig
        np.int64_t c1_count
        np.int64_t c2_count

        np.int64_t total_all
        int split, next_class

    initial_ent = binary_entropy(
        c1_traversed + c1_to_add,
        c2_traversed + c2_to_add,
    )

    total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

    bsf_ig = 0
    # actual observations in orderline
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        # optimistically add this class to left side first and other to right
        left_prop = (split + 1 + c1_to_add) / total_all
        ent_left = binary_entropy(c1_count + c1_to_add, c2_count)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = binary_entropy(c1_count, c2_count + c2_to_add)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = binary_entropy(
            c1_traversed - c1_count + c1_to_add,
            c2_traversed - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        if bsf_ig > worst_quality:
            return bsf_ig

    return bsf_ig



def calc_binary_ig(list orderline, np.int64_t c1, np.int64_t c2):

    cdef:
        np.float64_t initial_ent, left_prop, right_prop, ent_left, ent_right, ig
        np.float64_t bsf_ig
        np.int64_t c1_count
        np.int64_t c2_count

        np.int64_t total_all
        int split, next_class

    initial_ent = binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig


def binary_entropy(np.int64_t c1, np.int64_t c2):
    cdef:
        np.float64_t ent = 0
    if c1 != 0:
        ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
    if c2 != 0:
        ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
    return ent


def is_self_similar(tuple s1, tuple s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False



