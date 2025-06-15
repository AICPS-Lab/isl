import sys, os
sys.path.insert(0, os.getcwd())
from DSCLChecker.dscl import *
from functools import singledispatch
from scipy.ndimage import label
import numpy as np
from collections import deque
import random 
import itertools


@singledispatch
def qualitativescore(dscl_formula, image):
    raise TypeError("No qualitativescore for {} of class {}.".format(dscl_formula, dscl_formula.__class__))
    
    
@qualitativescore.register(IdAtomic)
def _(dscl_formula, image):
    return image[:, :, 0][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col] == dscl_formula.value
    
    
@qualitativescore.register(ClassAtomic)
def _(dscl_formula, image):
    return image[:, :, 1][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col] == dscl_formula.value
    

@qualitativescore.register(ProbAtomic)
def _(dscl_formula, image, tol=0):
    diff = robust_table[dscl_formula.comparator](image[:, :, 2][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col], dscl_formula.value)
    if dscl_formula.comparator in {"<=", ">="}:
        return diff >= -tol


@qualitativescore.register(IntensityAtomic)
def _(dscl_formula, image, tol=0):
    diff = robust_table[dscl_formula.comparator](image[:, :, 3][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col], dscl_formula.value)
    if dscl_formula.comparator in {"<=", ">="}:
        return diff >= -tol


@qualitativescore.register(RowAtomic)
def _(dscl_formula, _, tol=0):
    diff = robust_table[dscl_formula.comparator](dscl_formula.pixel_identifier.row, dscl_formula.value)
    if dscl_formula.comparator in {"<=", ">="}:
        return diff >= -tol


@qualitativescore.register(ColAtomic)
def _(dscl_formula, _, tol=0):
    diff = robust_table[dscl_formula.comparator](dscl_formula.pixel_identifier.col, dscl_formula.value)
    if dscl_formula.comparator in {"<=", ">="}:
        return diff >= -tol


@qualitativescore.register(Next)
def _(dscl_formula, image, tol=0):
    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for pixel_formula in pixel_formulas:
        curr_row, curr_col= pixel_formula.pixel_identifier.row, pixel_formula.pixel_identifier.col
        if dscl_formula.direction in pixel_dir_transform:
            new_row, new_col = pixel_dir_transform[dscl_formula.direction](curr_row, curr_col)
            if 0 <= new_row < image.shape[0] and 0 <= new_col < image.shape[1]:
                pixel_formula.pixel_identifier.row = new_row
                pixel_formula.pixel_identifier.col = new_col
            else:
                return False
            
    return qualitativescore(dscl_formula.subformula, image)
        
        
@qualitativescore.register(Exists)
def _(dscl_formula, image, tol=0):
    queried_regions = dscl_formula.region_identifier
    region_info = extract_component_bounds(image[:, :, 0])
    region_id, pixels = find_region_from_corners(region_info, queried_regions.p1, queried_regions.p2, queried_regions.p3, queried_regions.p4 )
    
    if not region_id: return False

    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for p in pixels:
        for pixel_formula in pixel_formulas:
            pixel_formula.pixel_identifier.row = p[0]
            pixel_formula.pixel_identifier.col = p[1]
        if qualitativescore(dscl_formula.subformula, image):
            return True
    return False


@qualitativescore.register(Forall)
def _(dscl_formula, image, tol=0):
    queried_regions = dscl_formula.region_identifier
    region_info = extract_component_bounds(image[:, :, 0])
    region_id, pixels = find_region_from_corners(region_info, queried_regions.p1, queried_regions.p2, queried_regions.p3, queried_regions.p4 )
    
    if not region_id: return False

    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for p in pixels:
        for pixel_formula in pixel_formulas:
            pixel_formula.pixel_identifier.row = p[0]
            pixel_formula.pixel_identifier.col = p[1]
        if not qualitativescore(dscl_formula.subformula, image):
            return False
    return True


@qualitativescore.register(WeakDistance)
def _(dscl_formula, image, tol=0):
    reference_corners = [
        dscl_formula.region_identifier.p1,
        dscl_formula.region_identifier.p2,
        dscl_formula.region_identifier.p3,
        dscl_formula.region_identifier.p4
    ]
    region_formulas = find_all_region_formulas(dscl_formula.subformula)
    region_info = extract_component_bounds(image[:, :, 0])

    rf_to_components = []
    for rf in region_formulas:
        prototype_corners = [
            rf.region_identifier.p1,
            rf.region_identifier.p2,
            rf.region_identifier.p3,
            rf.region_identifier.p4
        ]
        allowed_regions = explore_translated_regions_weak(
            prototype_corners, reference_corners,
            dscl_formula.direction, dscl_formula.step, image.shape[:2]
        )
        enclosed_components = find_enclosed_components(allowed_regions, region_info)
        # print(f"Prototype corners: {prototype_corners}")
        # print(f"Reference corners: {reference_corners}")
        # print(f"Allowed regions: {allowed_regions}")
        # print(f"Enclosed components: {enclosed_components}")
        if enclosed_components:  # WeakDistance: at least one valid region is enough
            rf_to_components.append((rf, enclosed_components))

    if not rf_to_components: return False

    component_combinations = itertools.product(*[comps for (_, comps) in rf_to_components])
    for combo in component_combinations:
        for (rf, _), region_id in zip(rf_to_components, combo):
            info = region_info[region_id]
            rf.region_identifier.p1 = PixelIdentifier(*info['top_left'])
            rf.region_identifier.p2 = PixelIdentifier(*info['top_right'])
            rf.region_identifier.p3 = PixelIdentifier(*info['bottom_left'])
            rf.region_identifier.p4 = PixelIdentifier(*info['bottom_right'])
        if qualitativescore(dscl_formula.subformula, image):
            return True
    return False


@qualitativescore.register(StrongDistance)
def _(dscl_formula, image, tol=0):
    reference_corners = [dscl_formula.region_identifier.p1, dscl_formula.region_identifier.p2, dscl_formula.region_identifier.p3, dscl_formula.region_identifier.p4]
    region_formulas = find_all_region_formulas(dscl_formula.subformula)
    region_info = extract_component_bounds(image[:, :, 0])
    
    # import logging
    # logger = logging.getLogger(__name__)

    rf_to_components = []
    for rf in region_formulas:
        prototype_corners = [rf.region_identifier.p1, rf.region_identifier.p2, rf.region_identifier.p3, rf.region_identifier.p4]
        allowed_regions = explore_translated_regions(prototype_corners, reference_corners, dscl_formula.direction, dscl_formula.step, image.shape[:2])
        enclosed_components = find_enclosed_components(allowed_regions, region_info)    # find subregions that are enclosed by the allowed regions
        # logger.debug(f"Enclosed components: {enclosed_components}")
        # print(f"Prototype corners: {prototype_corners}")
        # print(f"Reference corners: {reference_corners}")
        # print(f"Allowed regions: {allowed_regions}")
        # print(f"Enclosed components: {enclosed_components}")
        rf_to_components.append((rf, enclosed_components))

    component_combinations = itertools.product(*[comps for (_, comps) in rf_to_components])
    for combo in component_combinations:
        for (rf, _), region_id in zip(rf_to_components, combo):
            info = region_info[region_id]
            rf.region_identifier.p1 = PixelIdentifier(*info['top_left'])
            rf.region_identifier.p2 = PixelIdentifier(*info['top_right'])
            rf.region_identifier.p3 = PixelIdentifier(*info['bottom_left'])
            rf.region_identifier.p4 = PixelIdentifier(*info['bottom_right'])
        if qualitativescore(dscl_formula.subformula, image):
            return True
    return False


@qualitativescore.register(Negation)
def _(dscl_formula, image, tol=0):
    return not qualitativescore(dscl_formula.subformula, image)


@qualitativescore.register(And)
def _(dscl_formula, image, tol=0):
    return qualitativescore(dscl_formula.left, image) and qualitativescore(dscl_formula.right, image)


@qualitativescore.register(Or)
def _(dscl_formula, image, tol=0):
    return qualitativescore(dscl_formula.left, image) or qualitativescore(dscl_formula.right, image)


@qualitativescore.register(Implies)
def _(dscl_formula, image, tol=0):
    return (not qualitativescore(dscl_formula.left, image)) or qualitativescore(dscl_formula.right, image)


# begin quantitativescore functions 
@singledispatch
def quantitativescore(dscl_formula, image): 
    raise TypeError("No quantitativescore for {} of class {}.".format(dscl_formula, dscl_formula.__class__))
    
    
@quantitativescore.register(IdAtomic)
def _(dscl_formula, image):
    result = image[:, :, 0][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col] == dscl_formula.value
    return 1 if result else -1


@quantitativescore.register(ClassAtomic)
def _(dscl_formula, image):
    result = image[:, :, 1][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col] == dscl_formula.value
    return 1 if result else -1


@quantitativescore.register(ProbAtomic)
def _(dscl_formula, image):
    return robust_table[dscl_formula.comparator](image[:, :, 2][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col], dscl_formula.value)


@quantitativescore.register(IntensityAtomic)
def _(dscl_formula, image):
    return robust_table[dscl_formula.comparator](image[:, :, 3][dscl_formula.pixel_identifier.row, dscl_formula.pixel_identifier.col], dscl_formula.value)


@quantitativescore.register(RowAtomic)
def _(dscl_formula, image):
    return robust_table[dscl_formula.comparator](dscl_formula.pixel_identifier.row, dscl_formula.value)


@quantitativescore.register(ColAtomic)
def _(dscl_formula, image):
    return robust_table[dscl_formula.comparator](dscl_formula.pixel_identifier.col, dscl_formula.value)


@quantitativescore.register(Next)
def _(dscl_formula, image):
    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for pixel_formula in pixel_formulas:
        curr_row, curr_col= pixel_formula.pixel_identifier.row, pixel_formula.pixel_identifier.col
        if dscl_formula.direction in pixel_dir_transform:
            new_row, new_col = pixel_dir_transform[dscl_formula.direction](curr_row, curr_col)
            if 0 <= new_row < image.shape[0] and 0 <= new_col < image.shape[1]:
                pixel_formula.pixel_identifier.row = new_row
                pixel_formula.pixel_identifier.col = new_col
            else:
                return -np.inf  # out of bounds
    return quantitativescore(dscl_formula.subformula, image)


@quantitativescore.register(Exists)
def _(dscl_formula, image):
    queried_regions = dscl_formula.region_identifier
    region_info = extract_component_bounds(image[:, :, 0])
    region_id, pixels = find_region_from_corners(region_info, queried_regions.p1, queried_regions.p2, queried_regions.p3, queried_regions.p4 )
    
    if not region_id: return -np.inf

    max_score = -np.inf
    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for p in pixels:
        for pixel_formula in pixel_formulas:
            pixel_formula.pixel_identifier.row = p[0]
            pixel_formula.pixel_identifier.col = p[1]
        score = quantitativescore(dscl_formula.subformula, image)
        if score > max_score:
            max_score = score
    return max_score


@quantitativescore.register(Forall)
def _(dscl_formula, image):
    queried_regions = dscl_formula.region_identifier
    region_info = extract_component_bounds(image[:, :, 0])
    region_id, pixels = find_region_from_corners(region_info, queried_regions.p1, queried_regions.p2, queried_regions.p3, queried_regions.p4 )
    
    if not region_id: return -np.inf

    min_score = np.inf
    pixel_formulas = find_all_pixel_formulas(dscl_formula.subformula)
    for p in pixels:
        for pixel_formula in pixel_formulas:
            pixel_formula.pixel_identifier.row = p[0]
            pixel_formula.pixel_identifier.col = p[1]
        score = quantitativescore(dscl_formula.subformula, image)
        if score < min_score:
            min_score = score
    return min_score


@quantitativescore.register(StrongDistance)
def _(dscl_formula, image):
    reference_corners = [dscl_formula.region_identifier.p1, dscl_formula.region_identifier.p2, dscl_formula.region_identifier.p3, dscl_formula.region_identifier.p4]
    region_formulas = find_all_region_formulas(dscl_formula.subformula)
    region_info = extract_component_bounds(image[:, :, 0])

    rf_to_components = []
    for rf in region_formulas:
        prototype_corners = [rf.region_identifier.p1, rf.region_identifier.p2, rf.region_identifier.p3, rf.region_identifier.p4]
        allowed_regions = explore_translated_regions(prototype_corners, reference_corners, dscl_formula.direction, dscl_formula.step, image.shape[:2])
        enclosed_components = find_enclosed_components(allowed_regions, region_info)
        # print(f"Prototype corners: {prototype_corners}")
        # print(f"Reference corners: {reference_corners}")
        # print(f"Allowed regions: {allowed_regions}")
        # print(f"Enclosed components: {enclosed_components}")
        rf_to_components.append((rf, enclosed_components))

    component_combinations = itertools.product(*[comps for (_, comps) in rf_to_components])
    max_score = -np.inf
    for combo in component_combinations:
        for (rf, _), region_id in zip(rf_to_components, combo):
            info = region_info[region_id]
            rf.region_identifier.p1 = PixelIdentifier(*info['top_left'])
            rf.region_identifier.p2 = PixelIdentifier(*info['top_right'])
            rf.region_identifier.p3 = PixelIdentifier(*info['bottom_left'])
            rf.region_identifier.p4 = PixelIdentifier(*info['bottom_right'])
        score = quantitativescore(dscl_formula.subformula, image)
        if score > max_score:
            max_score = score
    return max_score


@quantitativescore.register(WeakDistance)
def _(dscl_formula, image):
    reference_corners = [dscl_formula.region_identifier.p1, dscl_formula.region_identifier.p2, dscl_formula.region_identifier.p3, dscl_formula.region_identifier.p4]
    region_formulas = find_all_region_formulas(dscl_formula.subformula)
    region_info = extract_component_bounds(image[:, :, 0])

    rf_to_components = []
    for rf in region_formulas:
        prototype_corners = [rf.region_identifier.p1, rf.region_identifier.p2, rf.region_identifier.p3, rf.region_identifier.p4]
        allowed_regions = explore_translated_regions_weak(prototype_corners, reference_corners, dscl_formula.direction, dscl_formula.step, image.shape[:2])
        enclosed_components = find_enclosed_components(allowed_regions, region_info)
        # print(f"Prototype corners: {prototype_corners}")
        # print(f"Reference corners: {reference_corners}")
        # print(f"Allowed regions: {allowed_regions}")
        # print(f"Enclosed components: {enclosed_components}")
        if enclosed_components:
            rf_to_components.append((rf, enclosed_components))

    if not rf_to_components:
        return -np.inf

    component_combinations = itertools.product(*[comps for (_, comps) in rf_to_components])
    max_score = -np.inf
    for combo in component_combinations:
        for (rf, _), region_id in zip(rf_to_components, combo):
            info = region_info[region_id]
            rf.region_identifier.p1 = PixelIdentifier(*info['top_left'])
            rf.region_identifier.p2 = PixelIdentifier(*info['top_right'])
            rf.region_identifier.p3 = PixelIdentifier(*info['bottom_left'])
            rf.region_identifier.p4 = PixelIdentifier(*info['bottom_right'])
        score = quantitativescore(dscl_formula.subformula, image)
        max_score = max(max_score, score)

    return max_score


@quantitativescore.register(Negation)
def _(dscl_formula, image):
    return -1*quantitativescore(dscl_formula.subformula, image)


@quantitativescore.register(And)
def _(dscl_formula, image):
    left_score = quantitativescore(dscl_formula.left, image)
    right_score = quantitativescore(dscl_formula.right, image)
    return min(left_score, right_score)


@quantitativescore.register(Or)
def _(dscl_formula, image):
    left_score = quantitativescore(dscl_formula.left, image)
    right_score = quantitativescore(dscl_formula.right, image)
    return max(left_score, right_score)


@quantitativescore.register(Implies)
def _(dscl_formula, image):
    return max(-1*quantitativescore(dscl_formula.left, image), quantitativescore(dscl_formula.right, image))


# helper functions
pixel_dir_transform = {"N": lambda r, c: (r-1, c), "S": lambda r, c: (r+1, c), "E": lambda r, c: (r, c-1), "W": lambda r, c: (r, c+1)}
robust_table = {"<":   lambda x, y: y - x, "<=":  lambda x, y: y - x, ">":   lambda x, y: x - y, ">=":  lambda x, y: x - y, "==":  lambda x, y: -abs(x - y), "!=":  lambda x, y: abs(x - y)}

def find_all_pixel_formulas(formula):
    results = []
    if hasattr(formula, "pixel_identifier"):
        results.append(formula)
    if hasattr(formula, "children"):
        for child in formula.children():
            results.extend(find_all_pixel_formulas(child))
    return results

def find_all_region_formulas(formula):
    results = []
    if hasattr(formula, "region_identifier"):
        results.append(formula)
    if hasattr(formula, "children"):
        for child in formula.children():
            results.extend(find_all_region_formulas(child))
    return results

def to_py_tuple(t):
    if hasattr(t, "__iter__") and len(t) == 2:
        return (int(t[0]), int(t[1]))
    elif hasattr(t, "row") and hasattr(t, "col"):
        return (int(t.row), int(t.col))
    elif hasattr(t, "x") and hasattr(t, "y"):
        return (int(t.x), int(t.y))
    else:
        raise TypeError(f"Unsupported type for corner: {t}")

def extract_component_bounds(image):
    structure_4 = np.array([[1,1,1], [1,1,1], [1,1,1]])
    region_info = {}
    region_id = 1

    for val in np.unique(image):
        mask = (image == val)
        labeled, num = label(mask, structure=structure_4)

        for comp_id in range(1, num + 1):
            coords = np.argwhere(labeled == comp_id)
            if coords.size == 0: continue
            rows, cols = coords[:, 0], coords[:, 1]
            top_row = rows.min()
            bottom_row = rows.max()
            top_candidates = coords[rows == top_row]
            bottom_candidates = coords[rows == bottom_row]
            region_info[region_id] = { "region_id": region_id, "value": int(val), "component": comp_id, "top_left": tuple(top_candidates[np.argmin(top_candidates[:, 1])]), "top_right": tuple(top_candidates[np.argmax(top_candidates[:, 1])]), "bottom_left": tuple(bottom_candidates[np.argmin(bottom_candidates[:, 1])]), "bottom_right": tuple(bottom_candidates[np.argmax(bottom_candidates[:, 1])]), "pixels": [tuple(p) for p in coords] }
            region_id += 1
            
    return region_info

def find_region_from_corners(region_info, top_left, top_right, bottom_left, bottom_right):
    target = tuple(to_py_tuple(x) for x in [top_left, top_right, bottom_left, bottom_right])
    for info in region_info.values():
        current = tuple(to_py_tuple(info[k]) for k in ["top_left", "top_right", "bottom_left", "bottom_right"])
        if current == target:
            return info['region_id'], info['pixels']
    return None, []

def identify_corners(corners):
    sorted_by_row = sorted(corners, key=lambda p: p.row)
    top_two = sorted(sorted_by_row[:2], key=lambda p: p.col)
    bot_two = sorted(sorted_by_row[2:], key=lambda p: p.col)
    return {'top_left': top_two[0],'top_right': top_two[1],'bottom_left': bot_two[0],'bottom_right': bot_two[1]}

# def explore_translated_regions(prototype_corners, reference_corners, direction, s, image_shape):
#     H, W = image_shape
#     proto_labeled = identify_corners(prototype_corners)
#     ref_labeled = identify_corners(reference_corners)

#     shift_row = shift_col = 0
#     step = s.step
#     shifted_regions = []

#     while True:
#         if direction == "N":
#             ref_top = ref_labeled['top_left'].row
#             proto_bottom = proto_labeled['bottom_left'].row
#             shift_row = ref_top - step - proto_bottom
#             shift_col = 0
#         elif direction == "S":
#             ref_bottom = ref_labeled['bottom_left'].row
#             proto_top = proto_labeled['top_left'].row
#             shift_row = ref_bottom + step - proto_top
#             shift_col = 0
#         elif direction == "W":
#             ref_left = ref_labeled['top_left'].col
#             proto_right = proto_labeled['top_right'].col
#             shift_row = 0
#             shift_col = ref_left - step - proto_right
#         elif direction == "E":
#             ref_right = ref_labeled['top_right'].col
#             proto_left = proto_labeled['top_left'].col
#             shift_row = 0
#             shift_col = ref_right + step - proto_left
#         else:
#             raise ValueError("Invalid direction")

#         def shifted(p):
#             return PixelIdentifier(p.row + shift_row, p.col + shift_col)

#         shifted_corners = {k: shifted(p) for k, p in proto_labeled.items()}
#         shifted_coords = [shifted_corners[k] for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']]
#         if any(p.row < 0 or p.col < 0 or p.row >= H or p.col >= W for p in shifted_coords): break

#         shifted_regions.append(shifted_corners)
#         step += 1
    
#     return shifted_regions


def get_bounds(corners):
        rows = [pt.row for pt in corners]
        cols = [pt.col for pt in corners]
        return min(rows), max(rows), min(cols), max(cols)
    
    
def explore_translated_regions(prototype_corners, reference_corners, direction, step, image_shape):
    H, W = image_shape
    shifted_regions = []

    def get_bounds(corners):
        rows = [p.row for p in corners]
        cols = [p.col for p in corners]
        return min(rows), max(rows), min(cols), max(cols)

    proto_top, proto_bot, proto_left, proto_right = get_bounds(prototype_corners)
    ref_top, ref_bot, ref_left, ref_right = get_bounds(reference_corners)

    proto_height = proto_bot - proto_top
    proto_width = proto_right - proto_left
    delta = step.step

    if direction == 'E':
        start_col = max(ref_right + delta, 0)
        for r0 in range(0, H - proto_height + 1):
            for c0 in range(start_col, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'W':
        print(f"Reference left: {ref_left}, Delta: {delta}, Proto width: {proto_width}")
        end_col = min(ref_left - delta, W)
        for r0 in range(0, H - proto_height + 1):
            for c0 in range(0, end_col - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'S':
        start_row = max(ref_bot + delta, 0)
        for r0 in range(start_row, H - proto_height + 1):
            for c0 in range(0, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'N':
        end_row = min(ref_top - delta, H)
        for r0 in range(0, end_row - proto_height + 1):
            for c0 in range(0, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    else:
        raise ValueError("Invalid direction. Must be one of {'N', 'S', 'E', 'W'}")

    return shifted_regions


def explore_translated_regions_weak(prototype_corners, reference_corners, direction, step, image_shape):
    H, W = image_shape
    shifted_regions = []

    def get_bounds(corners):
        rows = [p.row for p in corners]
        cols = [p.col for p in corners]
        return min(rows), max(rows), min(cols), max(cols)

    proto_top, proto_bot, proto_left, proto_right = get_bounds(prototype_corners)
    ref_top, ref_bot, ref_left, ref_right = get_bounds(reference_corners)

    proto_height = proto_bot - proto_top
    proto_width = proto_right - proto_left
    delta = step.step

    if direction == 'E':
        # Only constraint: left of prototype ≥ left of reference + delta
        min_col = max(ref_left + delta, 0)
        for r0 in range(0, H - proto_height + 1):
            for c0 in range(min_col, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'W':
        # Only constraint: right of prototype ≤ right of reference - delta
        max_col = min(ref_right - delta, W)
        for r0 in range(0, H - proto_height + 1):
            for c0 in range(0, max_col - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'S':
        # Only constraint: top of prototype ≥ top of reference + delta
        min_row = max(ref_top + delta, 0)
        for r0 in range(min_row, H - proto_height + 1):
            for c0 in range(0, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    elif direction == 'N':
        # Only constraint: bottom of prototype ≤ bottom of reference - delta
        max_row = min(ref_bot - delta, H)
        for r0 in range(0, max_row - proto_height + 1):
            for c0 in range(0, W - proto_width + 1):
                shifted_regions.append({
                    'top_left': PixelIdentifier(r0, c0),
                    'top_right': PixelIdentifier(r0, c0 + proto_width),
                    'bottom_left': PixelIdentifier(r0 + proto_height, c0),
                    'bottom_right': PixelIdentifier(r0 + proto_height, c0 + proto_width)
                })

    else:
        raise ValueError("Invalid direction. Must be one of {'N', 'S', 'E', 'W'}")

    return shifted_regions


# def explore_translated_regions_weak(prototype_corners, reference_corners, direction, s, image_shape):
#     H, W = image_shape
#     proto_labeled = identify_corners(prototype_corners)
#     ref_labeled = identify_corners(reference_corners)

#     shifted_regions = []

#     for shift in range(0, s.step + 1):
#         if direction == "N":
#             ref_top = ref_labeled['top_left'].row
#             proto_bottom = proto_labeled['bottom_left'].row
#             shift_row = ref_top - shift - proto_bottom
#             shift_col = 0
#         elif direction == "S":
#             ref_bottom = ref_labeled['bottom_left'].row
#             proto_top = proto_labeled['top_left'].row
#             shift_row = ref_bottom + shift - proto_top
#             shift_col = 0
#         elif direction == "W":
#             ref_left = ref_labeled['top_left'].col
#             proto_right = proto_labeled['top_right'].col
#             shift_row = 0
#             shift_col = ref_left - shift - proto_right
#         elif direction == "E":
#             ref_right = ref_labeled['top_right'].col
#             proto_left = proto_labeled['top_left'].col
#             shift_row = 0
#             shift_col = ref_right + shift - proto_left
#         else:
#             raise ValueError("Invalid direction")

#         def shifted(p): 
#             return PixelIdentifier(p.row + shift_row, p.col + shift_col)

#         shifted_corners = {k: shifted(p) for k, p in proto_labeled.items()}
#         shifted_coords = [shifted_corners[k] for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']]
#         if any(p.row < 0 or p.col < 0 or p.row >= H or p.col >= W for p in shifted_coords): continue

#         shifted_regions.append(shifted_corners)

#     return shifted_regions

def find_enclosed_components(allowed_regions, region_info):
    enclosed = []
    for region_id, info in region_info.items():
        region_corners = [info['top_left'],info['top_right'],info['bottom_left'],info['bottom_right']]
        for allowed in allowed_regions:
            rows = [pt.row for pt in allowed.values()]
            cols = [pt.col for pt in allowed.values()]
            rmin, rmax = min(rows), max(rows)
            cmin, cmax = min(cols), max(cols)
            if all(rmin <= r <= rmax and cmin <= c <= cmax for (r, c) in region_corners):
                enclosed.append(region_id)
                break
    return enclosed

# functions for testing
def generate_probability_matrix(rows, cols):
    return np.round(np.random.random((rows, cols)), 3)

def generate_color_matrix(rows, cols, options):
    return np.random.choice(options, size=(rows, cols))

def print_matrix(matrix):
    for row in matrix:
        print("  ".join(str(cell).ljust(6) for cell in row))
        
def grow_region(image, seed, label_value, max_size):
    rows, cols = image.shape
    queue = deque([seed])
    grown = 0

    while queue and grown < max_size:
        r, c = queue.popleft()
        if not (0 <= r < rows and 0 <= c < cols): continue
        if image[r, c] != 0: continue
        image[r, c] = label_value
        grown += 1

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            queue.append((r+dr, c+dc))

def generate_partitioned_image(rows=10, cols=10, num_regions=4, max_size=20):
    image = np.zeros((rows, cols), dtype=int)
    seeds = set()
    while len(seeds) < num_regions:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if image[r, c] == 0:
            seeds.add((r, c))
    for label_value, seed in enumerate(seeds, start=1):
        grow_region(image, seed, label_value, max_size)
        
    return image



    
def main():
    # Example usage
    rows, cols = 20, 20
    class_img = generate_color_matrix(rows, cols, ["1", "2", "3", "4", "5", "6", "7", "8"])
    id_image = generate_partitioned_image(rows, cols, num_regions=8)
    prob_img = generate_probability_matrix(rows, cols)
    print_matrix(class_img)
    print_matrix(id_image)
    print_matrix(prob_img)
    region_info = extract_component_bounds(id_image)
    augmented_image = np.zeros((rows, cols, 3), dtype=object)  # if mixed types
    augmented_image[:, :, 0] = id_image
    augmented_image[:, :, 1] = class_img
    augmented_image[:, :, 2] = prob_img
    
    logic = "( p(1,2).class == 1)"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "( ! p(1,2).class == 1)"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))

    logic = "  p(1,2).prob <=  0.5"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = " ! p(1,2).prob <=  0.5"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "  p(1,2).row <=  1"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, _))
    print(str(parsed)+":\t", quantitativescore(parsed, _))
    
    logic = "( p(2,2).row<=1)"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, _))
    print(str(parsed)+":\t", quantitativescore(parsed, _))
    
    logic = "(p(1,2).row <= 2 | ! p(1,2).row <= 2 )"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, _))
    print(str(parsed)+":\t", quantitativescore(parsed, _))
    
    logic = "(p(1,2).row <= 2 & ! p(1,2).row <= 2 )"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, _))
    print(str(parsed)+":\t", quantitativescore(parsed, _))
    
    logic = " (p(1,2).prob <=  0.5000 | p(1,2).prob >=  0.7) "
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "( ON p(1,2).prob >= 0.5 )"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "( ON (p(1,2).prob <=  0.7000 & p(1,2).prob >=  0.5)  )"
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    selected_key = random.choice(list(region_info.keys()))
    corners = region_info[selected_key]
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']
    bottom_right = corners['bottom_right']
    
    logic = "exists ({}, {}, {}, {}) p.id == 2".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "exists ({}, {}, {}, {}) p.prob >= 0.8".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "exists ({}, {}, {}, {}) ((ON p.prob >= 0.5 & p.class == 1) | p.id == 2 ) ".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "forall ({}, {}, {}, {}) p.id == 2".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "forall ({}, {}, {}, {}) ((ON p.prob >= 0.5 & p.class == 1) | p.id == 2 ) ".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "solidtriangle N [3] ({}, {}, {}, {}) ( exists ((0, 0), (0, 10), (10, 8), (8, 0)) p.prob >= 0.8)".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "solidtriangle N [3] ({}, {}, {}, {}) (forall ((0, 0), (0, 10), (10, 8), (8, 0)) p.prob >= 0.3)".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    logic = "solidtriangle N [3] ({}, {}, {}, {})  ( exists ((0, 0), (0, 10), (10, 8), (8, 0)) p.prob >= 0.8 | exists ((0, 0), (0, 10), (10, 8), (8, 0)) p.prob <= 0.9) ".format(top_left, top_right, bottom_left, bottom_right)
    parsed = parse(logic)
    print(str(parsed)+":\t", qualitativescore(parsed, augmented_image))
    print(str(parsed)+":\t", quantitativescore(parsed, augmented_image))
    
    print("Done.")