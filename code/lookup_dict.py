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
    0: "n02124075",
    "n02124075": 0,
    1: "n02504458",
    "n02504458": 1,
    2: "n02389026",
    "n02389026": 2,
    3: "n02492035",
    "n02492035": 3,
    4: "n02510455",
    "n02510455": 4,
    5: "n02106662",
    "n02106662": 5,
    6: "n04086273",
    "n04086273": 6,
    7: "n03452741",
    "n03452741": 7,
    8: "n02690373",
    "n02690373": 8,
    9: "n02951358",
    "n02951358": 9,
    10: "n03773504",
    "n03773504": 10,
    11: "n03792782",
    "n03792782": 11,
    12: "n03272562",
    "n03272562": 12,
    13: "n03100240",
    "n03100240": 13,
    14: "n03376595",
    "n03376595": 14,
    15: "n03982430",
    "n03982430": 15,
    16: "n07753592",
    "n07753592": 16,
    17: "n03272010",
    "n03272010": 17,
    18: "n11939491",
    "n11939491": 18,
    19: "n02607072",
    "n02607072": 19,
    20: "n03197337",
    "n03197337": 20,
    21: "n04044716",
    "n04044716": 21,
    22: "n03180011",
    "n03180011": 22,
    23: "n03590841",
    "n03590841": 23,
    24: "n02281787",
    "n02281787": 24,
    25: "n03584829",
    "n03584829": 25,
    26: "n03297495",
    "n03297495": 26,
    27: "n03792972",
    "n03792972": 27,
    28: "n03877472",
    "n03877472": 28,
    29: "n04120489",
    "n04120489": 29,
    30: "n03445777",
    "n03445777": 30,
    31: "n03709823",
    "n03709823": 31,
    32: "n02906734",
    "n02906734": 32,
    33: "n03775071",
    "n03775071": 33,
    34: "n02992529",
    "n02992529": 34,
    35: "n03888257",
    "n03888257": 35,
    36: "n07873807",
    "n07873807": 36,
    37: "n04069434",
    "n04069434": 37,
    38: "n13054560",
    "n13054560": 38,
    39: "n03063599",
    "n03063599": 39,
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
