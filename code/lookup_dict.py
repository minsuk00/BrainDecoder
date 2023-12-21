id_to_name = {
    "n02124075": "Egyptian_cat",
    "n02504458": "African_elephant",
    "n02389026": "sorrel",
    "n02492035": "capuchin",
    "n02510455": "giant_panda",
    "n02106662": "German_shepherd",
    "n04086273": "revolver",
    "n03452741": "grand_piano",
    "n02690373": "airliner",
    "n02951358": "canoe",
    "n03773504": "missile",
    "n03792782": "mountain_bike",
    "n03272562": "electric_locomotive",
    "n03100240": "convertible",
    "n03376595": "folding_chair",
    "n03982430": "pool_table",
    "n07753592": "banana",
    "n03272010": "electric_guitar",
    "n11939491": "daisy",
    "n02607072": "anemone_fish",
    "n03197337": "digital_watch",
    "n04044716": "radio_telescope",
    "n03180011": "desktop_computer",
    "n03590841": "jack-o'-lantern",
    "n02281787": "lycaenid",
    "n03584829": "iron",
    "n03297495": "espresso_maker",
    "n03792972": "mountain_tent",
    "n03877472": "pajama",
    "n04120489": "running_shoe",
    "n03445777": "golf_ball",
    "n03709823": "mailbag",
    "n02906734": "broom",
    "n03775071": "mitten",
    "n02992529": "cellular_telephone",
    "n03888257": "parachute",
    "n07873807": "pizza",
    "n04069434": "reflex_camera",
    "n13054560": "bolete",
    "n03063599": "coffee_mug",
}

idx_to_id = {
    0: "n02389026",
    "n02389026": 0,
    1: "n03888257",
    "n03888257": 1,
    2: "n03584829",
    "n03584829": 2,
    3: "n02607072",
    "n02607072": 3,
    4: "n03297495",
    "n03297495": 4,
    5: "n03063599",
    "n03063599": 5,
    6: "n03792782",
    "n03792782": 6,
    7: "n04086273",
    "n04086273": 7,
    8: "n02510455",
    "n02510455": 8,
    9: "n11939491",
    "n11939491": 9,
    10: "n02951358",
    "n02951358": 10,
    11: "n02281787",
    "n02281787": 11,
    12: "n02106662",
    "n02106662": 12,
    13: "n04120489",
    "n04120489": 13,
    14: "n03590841",
    "n03590841": 14,
    15: "n02992529",
    "n02992529": 15,
    16: "n03445777",
    "n03445777": 16,
    17: "n03180011",
    "n03180011": 17,
    18: "n02906734",
    "n02906734": 18,
    19: "n07873807",
    "n07873807": 19,
    20: "n03773504",
    "n03773504": 20,
    21: "n02492035",
    "n02492035": 21,
    22: "n03982430",
    "n03982430": 22,
    23: "n03709823",
    "n03709823": 23,
    24: "n03100240",
    "n03100240": 24,
    25: "n03376595",
    "n03376595": 25,
    26: "n03877472",
    "n03877472": 26,
    27: "n03775071",
    "n03775071": 27,
    28: "n03272010",
    "n03272010": 28,
    29: "n04069434",
    "n04069434": 29,
    30: "n03452741",
    "n03452741": 30,
    31: "n03792972",
    "n03792972": 31,
    32: "n07753592",
    "n07753592": 32,
    33: "n13054560",
    "n13054560": 33,
    34: "n03197337",
    "n03197337": 34,
    35: "n02504458",
    "n02504458": 35,
    36: "n02690373",
    "n02690373": 36,
    37: "n03272562",
    "n03272562": 37,
    38: "n04044716",
    "n04044716": 38,
    39: "n02124075",
    "n02124075": 39,
}


def batch_idx_to_id(batch_indices):
    """converts a tensor of indicies to a list of ids

    Args:
        batch_indices (tensor): indices from dataset

    Returns:
        list: id list
    """

    id_list = []

    for idx in batch_indices:
        id_list.append(idx_to_id[idx])

    return id_list


def batch_id_to_name(batch_ids):
    """converst a list of ids to a list of names

    Args:
        batch_ids (list): list of ids from batch_idx_to_id

    Returns:
        list: list of name
    """

    name_list = []

    for id in batch_ids:
        name_list.append(id_to_name[id])

    return name_list
