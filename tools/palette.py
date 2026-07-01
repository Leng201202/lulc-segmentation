import numpy as np

# IRSAMap: https://github.com/ucas-dlg/IRSAMap
# Modified to 9 classes (8 + background):
# - Merged river/lakes/sea → water_body
# - Merged road_area/road_line → road

CLASS_NAMES = [
    "background",
    "farmland",
    "tree",
    "grass",
    "water_body",
    "building",
    "road",
    "sport",
    "bareland",
]

# ref-code 1-11 -> class index (with merged water bodies and roads)
REF_CODE_TO_CLASS = {
    0: 0,
    1: 1,  # farmland
    2: 2,  # tree
    3: 3,  # grass
    4: 4,  # river → water_body
    5: 4,  # lakes → water_body
    6: 4,  # sea → water_body
    7: 5,  # building
    8: 6,  # road_area → road
    9: 6,  # road_line → road
    10: 7,  # sport
    11: 8,  # bareland
}

# Official second-level category codes (SegLabel_rvwsb / SegLabel_vwsbr).
CATEGORY_CODE_TO_CLASS = {
    10: 1,   # farmland
    11: 2,   # tree
    12: 3,   # grass
    21: 4,   # river → water_body
    22: 4,   # lakes → water_body
    23: 4,   # sea → water_body
    31: 5,   # building
    32: 6,   # road_area → road
    33: 6,   # road_line → road
    34: 7,   # sport
    40: 8,   # bareland
}

# Official RGB colors (ref-code 1-11).
RGB_TO_CLASS = {
    (255, 253, 145): 1,
    (32, 216, 109): 2,
    (1, 252, 119): 3,
    (20, 197, 246): 4,  # river → water_body
    (20, 185, 246): 4,  # lakes → water_body
    (20, 197, 232): 4,  # sea → water_body
    (210, 75, 97): 5,
    (255, 200, 1): 6,  # road_area → road (using road_area color)
    (192, 1, 255): 6,  # road_line → road
    (255, 156, 95): 7,
    (204, 181, 206): 8,
}

COLOR_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 253, 145],
        [32, 216, 109],
        [1, 252, 119],
        [20, 197, 246],  # water_body
        [210, 75, 97],
        [255, 200, 1],  # road
        [255, 156, 95],
        [204, 181, 206],
    ],
    dtype=np.uint8,
)


def rgb_mask_to_class(mask_rgb: np.ndarray, tolerance: int = 5) -> np.ndarray:
    h, w = mask_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.int64)
    rgb = mask_rgb[..., :3].astype(np.int16)

    for color, class_id in RGB_TO_CLASS.items():
        color_arr = np.array(color, dtype=np.int16)
        diff = np.abs(rgb - color_arr).max(axis=-1)
        class_map[diff <= tolerance] = class_id

    return class_map


def ref_code_mask_to_class(mask: np.ndarray) -> np.ndarray:
    class_map = mask.astype(np.int64)
    return np.clip(class_map, 0, len(CLASS_NAMES) - 1)


def category_code_mask_to_class(mask: np.ndarray) -> np.ndarray:
    class_map = np.zeros(mask.shape, dtype=np.int64)
    for code, class_id in CATEGORY_CODE_TO_CLASS.items():
        class_map[mask == code] = class_id
    return class_map


def mask_to_class_indices(mask: np.ndarray, encoding: str = "category_code") -> np.ndarray:
    if mask.ndim == 3:
        return rgb_mask_to_class(mask)

    if encoding == "ref_code":
        return ref_code_mask_to_class(mask)
    if encoding == "category_code":
        return category_code_mask_to_class(mask)
    if encoding == "rgb":
        raise ValueError("RGB encoding requires a 3-channel mask array.")

    raise ValueError(f"Unknown label encoding: {encoding}")


def class_indices_to_color(class_map: np.ndarray) -> np.ndarray:
    class_map = np.clip(class_map, 0, len(COLOR_PALETTE) - 1)
    return COLOR_PALETTE[class_map]
