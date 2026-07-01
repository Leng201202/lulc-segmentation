"""Label mappings for IRSAMap and LoveDA support experiments."""

CLASS_NAMES = [
    "Farmland",
    "Tree",
    "Grass",
    "Water",
    "Building",
    "Road",
    "Sport",
    "Bareland",
]

NUM_CLASSES = 8
IGNORE_INDEX = 255

# Visualization colors (RGB) aligned with IRSAMap palette.
CLASS_COLORS = {
    0: (255, 253, 145),  # Farmland
    1: (32, 216, 109),   # Tree
    2: (1, 252, 119),    # Grass
    3: (20, 197, 246),   # Water
    4: (210, 75, 97),    # Building
    5: (255, 200, 1),    # Road
    6: (255, 156, 95),   # Sport
    7: (204, 181, 206),  # Bareland
}

IRSAMAP_RGB_TO_CLASS = {
    (255, 253, 145): 0,
    (32, 216, 109): 1,
    (1, 252, 119): 2,
    (20, 197, 246): 3,
    (20, 185, 246): 3,
    (20, 197, 232): 3,
    (210, 75, 97): 4,
    (255, 200, 1): 5,
    (192, 1, 255): 5,
    (255, 156, 95): 6,
    (204, 181, 206): 7,
}

LOVEDA_TO_IRSAMAP_SUPPORT = {
    0: 255,
    1: 255,
    2: 255,
    3: 5,
    4: 3,
    5: 7,
    6: 1,
    7: 0,
}

LOVEDA_TO_IRSAMAP_FULL_SUPPORT = {
    0: 255,
    1: 255,
    2: 4,
    3: 5,
    4: 3,
    5: 7,
    6: 1,
    7: 0,
}

LOVEDA_MAPPINGS = {
    "targeted": LOVEDA_TO_IRSAMAP_SUPPORT,
    "full": LOVEDA_TO_IRSAMAP_FULL_SUPPORT,
}
