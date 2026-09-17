from src.detection_postprocess import clean_detections, dedupe_key, is_structural, normalize_label


def test_normalize_strips_counts():
    assert normalize_label("Glass Storage Jars (2)") == "glass storage jars"
    assert normalize_label("bare trees (several)") == "bare trees"
    assert normalize_label("3 wooden cutting boards") == "wooden cutting boards"
    assert normalize_label("  Sofa  ") == "sofa"


def test_dedupe_key_merges_plurals_and_case():
    assert dedupe_key("Chairs") == dedupe_key("chair")
    assert dedupe_key("Bookshelves") == dedupe_key("bookshelf") or dedupe_key("Bookshelves") == "bookshelve"
    assert dedupe_key("dining chair") != dedupe_key("office chair")


def test_runaway_vlm_output_is_collapsed():
    objs = [{"id": 1, "label": "desk", "description": "wooden", "expected_instances": "single"}]
    objs += [{"id": i, "label": "tool", "description": "red and blue holders",
              "expected_instances": "single"} for i in range(2, 110)]
    out = clean_detections(objs, verbose=False)
    assert [o["label"] for o in out] == ["desk", "tool"]
    # 108 repeats of the same label is evidence of many instances
    assert out[1]["expected_instances"] == "multiple"
    assert [o["id"] for o in out] == [1, 2]


def test_people_and_structure_are_removed_but_compounds_survive():
    objs = [{"label": "person"}, {"label": "wall"}, {"label": "floor"},
            {"label": "wall clock"}, {"label": "floor lamp"}, {"label": "woman"},
            {"label": "window"}, {"label": "window blind"}]
    out = clean_detections(objs, verbose=False)
    assert [o["label"] for o in out] == ["wall clock", "floor lamp", "window blind"]
    assert is_structural("Ceiling") and not is_structural("ceiling fan")


def test_bbox_validation_and_clamping():
    objs = [{"label": "a", "bbox_2d": [10, 10, 5, 5]},          # inverted
            {"label": "b", "bbox_2d": [0, 0, 2000, 50]},        # clamps to image
            {"label": "c", "bbox_2d": "nope"},
            {"label": "d", "bbox_2d": [1, 2, 3, 4]}]
    out = clean_detections(objs, image_size=(640, 480), verbose=False)
    by = {o["label"]: o["bbox_2d"] for o in out}
    assert by["a"] is None and by["c"] is None
    assert by["b"] == [0.0, 0.0, 640.0, 50.0]
    assert by["d"] == [1.0, 2.0, 3.0, 4.0]


def test_cap_and_malformed():
    objs = [{"label": f"thing {i}"} for i in range(60)] + ["garbage", None, {"label": ""}]
    out = clean_detections(objs, max_objects=40, verbose=False)
    assert len(out) == 40 and out[-1]["id"] == 40


def test_parts_are_dropped_but_whole_objects_kept():
    objs = [{"label": "handle"}, {"label": "drawer"}, {"label": "cabinet"}, {"label": "door handle"},
            {"label": "light switch"}, {"label": "table lamp"}, {"label": "wheel"}, {"label": "wheelbarrow"}]
    out = clean_detections(objs, verbose=False)
    assert [o["label"] for o in out] == ["cabinet", "table lamp", "wheelbarrow"]
