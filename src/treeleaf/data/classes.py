"""Официальное соответствие ID файлов и классов Flavia."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlaviaClass:
    # Номер класса из оригинальной таблицы Flavia (в ней пропущен №13).
    source_label: int
    # Непрерывный индекс для обучения нейросети: 0..31.
    class_index: int
    slug: str
    scientific_name: str
    common_name: str
    image_id_start: int
    image_id_end: int


# Диапазоны взяты из официальной таблицы Flavia:
# https://flavia.sourceforge.net/
_RAW_CLASSES = [
    (1,  "pubescent_bamboo",          "Phyllostachys edulis (Carr.) Houz.",              "pubescent bamboo",          1001, 1059),
    (2,  "chinese_horse_chestnut",    "Aesculus chinensis",                              "Chinese horse chestnut",    1060, 1122),
    (3,  "anhui_barberry",            "Berberis anhweiensis Ahrendt",                    "Anhui Barberry",            1552, 1616),
    (4,  "chinese_redbud",            "Cercis chinensis",                                "Chinese redbud",            1123, 1194),
    (5,  "true_indigo",               "Indigofera tinctoria L.",                         "true indigo",               1195, 1267),
    (6,  "japanese_maple",            "Acer palmatum",                                   "Japanese maple",            1268, 1323),
    (7,  "nanmu",                     "Phoebe nanmu (Oliv.) Gamble",                     "Nanmu",                     1324, 1385),
    (8,  "castor_aralia",             "Kalopanax septemlobus (Thunb. ex A.Murr.) Koidz.", "castor aralia",             1386, 1437),
    (9,  "chinese_cinnamon",          "Cinnamomum japonicum Sieb.",                      "Chinese cinnamon",          1497, 1551),
    (10, "goldenrain_tree",           "Koelreuteria paniculata Laxm.",                   "goldenrain tree",           1438, 1496),
    (11, "big_fruited_holly",         "Ilex macrocarpa Oliv.",                           "Big-fruited Holly",         2001, 2050),
    (12, "japanese_cheesewood",       "Pittosporum tobira (Thunb.) Ait. f.",             "Japanese cheesewood",       2051, 2113),
    (14, "wintersweet",               "Chimonanthus praecox L.",                         "wintersweet",               2114, 2165),
    (15, "camphortree",               "Cinnamomum camphora (L.) J. Presl",               "camphortree",               2166, 2230),
    (16, "japan_arrowwood",           "Viburnum awabuki K.Koch",                         "Japan Arrowwood",           2231, 2290),
    (17, "sweet_osmanthus",           "Osmanthus fragrans Lour.",                        "sweet osmanthus",           2291, 2346),
    (18, "deodar",                    "Cedrus deodara (Roxb.) G. Don",                    "deodar",                    2347, 2423),
    (19, "ginkgo",                    "Ginkgo biloba L.",                                "ginkgo, maidenhair tree",   2424, 2485),
    (20, "crape_myrtle",              "Lagerstroemia indica (L.) Pers.",                 "Crape myrtle",              2486, 2546),
    (21, "oleander",                  "Nerium oleander L.",                              "oleander",                  2547, 2612),
    (22, "yew_plum_pine",             "Podocarpus macrophyllus (Thunb.) Sweet",          "yew plum pine",             2616, 2675),
    (23, "japanese_flowering_cherry", "Prunus serrulata Lindl. var. lannesiana auct.",   "Japanese Flowering Cherry", 3001, 3055),
    (24, "glossy_privet",             "Ligustrum lucidum Ait. f.",                       "Glossy Privet",             3056, 3110),
    (25, "chinese_toon",              "Toona sinensis M. Roem.",                         "Chinese Toon",              3111, 3175),
    (26, "peach",                     "Prunus persica (L.) Batsch",                      "peach",                     3176, 3229),
    (27, "ford_woodlotus",            "Manglietia fordiana Oliv.",                       "Ford Woodlotus",            3230, 3281),
    (28, "trident_maple",             "Acer buergerianum Miq.",                          "trident maple",             3282, 3334),
    (29, "beales_barberry",           "Mahonia bealei (Fortune) Carr.",                  "Beale's barberry",          3335, 3389),
    (30, "southern_magnolia",         "Magnolia grandiflora L.",                         "southern magnolia",         3390, 3446),
    (31, "canadian_poplar",           "Populus ×canadensis Moench",                      "Canadian poplar",           3447, 3510),
    (32, "chinese_tulip_tree",        "Liriodendron chinense (Hemsl.) Sarg.",            "Chinese tulip tree",        3511, 3563),
    (33, "tangerine",                 "Citrus reticulata Blanco",                        "tangerine",                 3566, 3621),
]

FLAVIA_CLASSES: tuple[FlaviaClass, ...] = tuple(
    FlaviaClass(
        source_label=row[0],
        class_index=index,
        slug=row[1],
        scientific_name=row[2],
        common_name=row[3],
        image_id_start=row[4],
        image_id_end=row[5],
    )
    for index, row in enumerate(_RAW_CLASSES)
)

if len(FLAVIA_CLASSES) != 32:
    raise RuntimeError(f"Ожидалось 32 класса Flavia, получено {len(FLAVIA_CLASSES)}")


def get_class_by_image_id(image_id: int) -> FlaviaClass | None:
    """Возвращает класс по числовому ID изображения (например, 1001)."""
    for item in FLAVIA_CLASSES:
        if item.image_id_start <= image_id <= item.image_id_end:
            return item
    return None


def get_class_by_index(class_index: int) -> FlaviaClass:
    """Возвращает класс по непрерывному индексу нейросети."""
    if not 0 <= class_index < len(FLAVIA_CLASSES):
        raise IndexError(f"Некорректный class_index: {class_index}")
    return FLAVIA_CLASSES[class_index]
