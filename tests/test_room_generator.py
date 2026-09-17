import numpy as np

from src import room_generator as rg


def test_floor_keyword_matching_is_token_based():
    yes = ["chair", "office chair", "coffee table", "floor lamp", "potted plant", "trash bin",
           "bookshelf", "refrigerator", "3d printer stand", "ottoman", "king bed"]
    no = ["table lamp", "tablet", "bedside lamp", "microwave oven", "desk lamp", "wall cabinet",
          "ceiling fan", "wall shelf", "pendant light", "mirror", "painting", "bedding",
          "desktop computer", "chair cushion"]
    assert all(rg.is_floor_supported(l) for l in yes), [l for l in yes if not rg.is_floor_supported(l)]
    assert not any(rg.is_floor_supported(l) for l in no), [l for l in no if rg.is_floor_supported(l)]


def _box(id_, label, mn, mx, conf=0.9):
    return ({"id": id_, "label": label, "confidence": conf,
             "translation": [0.0, 0.0, 0.0], "rotation_quaternion": [1, 0, 0, 0], "scale": [1, 1, 1]},
            {"min": list(map(float, mn)), "max": list(map(float, mx))})


def test_support_graph_finds_lamp_on_table_not_chair_under_table():
    table, tab = _box(1, "table", (0, 0.0, 0), (1.0, 0.75, 1.0))
    lamp, lab = _box(2, "table lamp", (0.3, 0.78, 0.3), (0.5, 1.2, 0.5))
    plant, pab = _box(3, "potted plant", (0.6, 0.70, 0.6), (0.8, 1.0, 0.8))   # slightly sunk into the top
    chair, cab = _box(4, "chair", (0.2, 0.0, -0.3), (0.7, 0.9, 0.4))           # tucked under, back rises above
    results = [table, lamp, plant, chair]
    aabbs = {1: tab, 2: lab, 3: pab, 4: cab}
    sup = rg.find_supports(results, aabbs, scene_h=1.5)
    assert sup == {2: 1, 3: 1}


def test_snap_uses_supports_and_floor():
    table, tab = _box(1, "table", (0, 0.1, 0), (1.0, 0.85, 1.0))          # floats 10 cm
    lamp, lab = _box(2, "table lamp", (0.3, 1.0, 0.3), (0.5, 1.4, 0.5))    # floats above the table
    art, aab = _box(3, "painting", (0.2, 1.5, 1.0), (0.8, 1.9, 1.05))       # wall art: leave alone
    results = [table, lamp, art]
    aabbs = {1: tab, 2: lab, 3: aab}
    bounds = rg.compute_scene_bounds(aabbs)
    sup = rg.find_supports(results, aabbs, scene_h=1.9)
    n = rg.snap_ground_objects(results, aabbs, bounds, floor_y=0.0, supports=sup)
    assert n == 2
    assert np.isclose(aabbs[1]["min"][1], 0.0) and np.isclose(table["translation"][1], -0.1)
    assert np.isclose(aabbs[2]["min"][1], aabbs[1]["max"][1])      # lamp sits on the table top
    assert art["snapped"] is False and aabbs[3]["min"][1] == 1.5


def test_collisions_skip_stacked_and_tucked_objects_but_fix_real_overlap():
    table, tab = _box(1, "table", (0, 0, 0), (1.0, 0.75, 1.0))
    lamp, lab = _box(2, "table lamp", (0.3, 0.75, 0.3), (0.5, 1.2, 0.5))
    chair, cab = _box(3, "chair", (0.2, 0.0, -0.3), (0.7, 0.9, 0.4))        # 57% of footprint under table
    sofa_a, sab = _box(4, "sofa", (2.0, 0, 0), (3.0, 0.8, 1.0), conf=0.9)
    sofa_b, sbb = _box(5, "sofa", (2.1, 0, 0.1), (3.1, 0.8, 1.1), conf=0.5)  # nearly coincident duplicate
    results = [table, lamp, chair, sofa_a, sofa_b]
    aabbs = {1: tab, 2: lab, 3: cab, 4: sab, 5: sbb}
    sup = rg.find_supports(results, aabbs, scene_h=1.2)
    pushes = rg.resolve_xz_collisions(results, aabbs, supports=sup)
    assert pushes >= 1
    assert lamp["translation"][0] == 0.0 and lamp["translation"][2] == 0.0      # not pushed off the table
    assert chair["translation"] == [0.0, 0.0, 0.0]                              # tucked chair stays
    # the low-confidence sofa moved more than the high-confidence one
    assert abs(sofa_b["translation"][0]) + abs(sofa_b["translation"][2]) > \
        abs(sofa_a["translation"][0]) + abs(sofa_a["translation"][2])
    # pushes are capped: nobody flew across the room
    assert all(abs(v) <= 0.5 for r in results for v in r["translation"])


def test_upright_correct_only_touches_near_upright():
    up = {"id": 1, "label": "x", "rotation_quaternion": [0.7071, -0.7071, 0, 0]}     # R_x(-90°): upright
    tilted = {"id": 2, "label": "y", "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]}     # local +Z along world +Z
    n = rg.upright_correct([up, tilted])
    assert n == 1 and up["upright_corrected"] and not tilted["upright_corrected"]
    assert tilted["rotation_quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert abs(tilted["tilt_deg"] - 90) < 1e-6


def test_hanging_objects_are_never_snapped_or_supported():
    sofa, sab = _box(1, "sofa", (0, 0, 0), (2.0, 0.8, 0.9))
    lamp, lab = _box(2, "pendant lamp", (0.8, 0.85, 0.3), (1.2, 1.3, 0.7))   # bottom 5 cm above sofa top
    results = [sofa, lamp]
    aabbs = {1: sab, 2: lab}
    assert rg.find_supports(results, aabbs, scene_h=2.5) == {}
    bounds = rg.compute_scene_bounds(aabbs)
    rg.snap_ground_objects(results, aabbs, bounds, floor_y=0.0, supports={})
    assert lamp["snapped"] is False and lamp["translation"] == [0.0, 0.0, 0.0]
    assert rg.is_hanging("ceiling fan") and rg.is_hanging("curtains") and not rg.is_hanging("floor lamp")


def test_clamp_to_room_only_enforces_detected_walls_and_caps_push():
    sofa, sab = _box(1, "sofa", (0.0, 0, 2.2), (2.0, 0.8, 3.1))      # pokes 0.3 through z_max=2.8
    far, fab = _box(2, "lamp", (0.0, 0, 5.0), (0.2, 1.0, 5.2))        # 2.2 beyond: capped push
    results = [sofa, far]
    aabbs = {1: sab, 2: fab}
    n = rg.clamp_to_room(results, aabbs, [-1, 0, 0], [3, 2.5, 2.8],
                         {"z_max": "wall(5000)", "x_min": "extent", "x_max": "extent", "z_min": "extent"})
    assert n == 2
    assert np.isclose(sab["max"][2], 2.8) and np.isclose(sofa["translation"][2], -0.3)
    assert np.isclose(fab["max"][2], 5.2 - 0.35 * 0.2)                # capped at 35 % of its extent
    # x sides are "extent": nothing happens even though the sofa is at x_min-... boundary
    assert sofa["translation"][0] == 0.0


def test_pillows_inside_sofa_box_are_left_alone():
    sofa, sab = _box(1, "sofa", (0.0, 0.0, 1.2), (2.0, 0.9, 1.9))
    pillow, pab = _box(2, "pillow", (0.3, 0.45, 1.6), (0.7, 0.85, 1.8))     # on the seat, under the back top
    pillow2, p2b = _box(3, "pillow", (1.2, 0.45, 1.6), (1.6, 0.85, 1.8))
    results = [sofa, pillow, pillow2]
    aabbs = {1: sab, 2: pab, 3: p2b}
    sup = rg.find_supports(results, aabbs, scene_h=2.5)
    assert 2 not in sup and 3 not in sup                       # support test cannot see them ...
    con = rg.find_contained(results, aabbs, sup)
    assert con == {2: 1, 3: 1}                                 # ... containment can
    bounds = rg.compute_scene_bounds(aabbs)
    rg.snap_ground_objects(results, aabbs, bounds, floor_y=0.0, supports=sup, contained=con)
    assert pillow["translation"][1] == 0.0 and pillow["contained_in"] == 1   # not dropped to the floor
    pushes = rg.resolve_xz_collisions(results, aabbs, supports=sup, contained=con)
    assert pushes == 0
    assert pillow["translation"] == [0.0, 0.0, 0.0] and pillow2["translation"] == [0.0, 0.0, 0.0]


def test_clamp_raises_objects_sunk_below_a_fitted_floor():
    chair, cab = _box(1, "chair", (0, -0.12, 0), (0.5, 0.8, 0.5))          # 12 cm under the floor
    results = [chair]
    aabbs = {1: cab}
    n = rg.clamp_to_room(results, aabbs, [-1, 0.0, -1], [3, 2.5, 3], {"y_min": "floor"})
    assert n == 1 and np.isclose(cab["min"][1], 0.0) and np.isclose(chair["translation"][1], 0.12)
