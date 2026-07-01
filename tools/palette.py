import numpy as np

# IRSAMap: https://github.com/ucas-dlg/IRSAMap
# Paper TABLE III — 11 land-cover classes + background (class index 0).

CLASS_NAMES = [
    "background",
    "farmland",
    "tree",
    "grass",
    "river",
    "lakes",
    "sea",
    "building",
    "road_area",
    "road_line",
    "sport",
    "bareland",
]

# ref-code 1-11 -> class index 1-11 (SegLabel_*_0-11 rasters).
REF_CODE_TO_CLASS = {i: i for i in range(12)}

# Official second-level category codes (SegLabel_rvwsb / SegLabel_vwsbr).
CATEGORY_CODE_TO_CLASS = {
    10: 1,   # farmland
    11: 2,   # tree
    12: 3,   # grass
    21: 4,   # river
    22: 5,   # lakes
    23: 6,   # sea
    31: 7,   # building
    32: 8,   # road_area
    33: 9,   # road_line
    34: 10,  # sport
    40: 11,  # bareland
}

# Official RGB colors (ref-code 1-11).
RGB_TO_CLASS = {
    (255, 253, 145): 1,
    (32, 216, 109): 2,
    (1, 252, 119): 3,
    (20, 197, 246): 4,
    (20, 185, 246): 5,
    (20, 197, 232): 6,
    (210, 75, 97): 7,
    (255, 200, 1): 8,
    (192, 1, 255): 9,
    (255, 156, 95): 10,
    (204, 181, 206): 11,
}

COLOR_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 253, 145],
        [32, 216, 109],
        [1, 252, 119],
        [20, 197, 246],
        [20, 185, 246],
        [20, 197, 232],
        [210, 75, 97],
        [255, 200, 1],
        [192, 1, 255],
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
