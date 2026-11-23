"""A detection that is just a patch of the surface around it is left to the room."""
import numpy as np

from src.scene_assembly import indistinct

H = W = 120


def wall(colour=(200, 190, 180), noise=4, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(np.array(colour, float) + rng.normal(0, noise, (H, W, 3)), 0, 255).astype(np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([xs / W * 3.0, 2.5 - ys / H * 2.5, np.full((H, W), 4.0)], -1)   # a wall 4 m away
    return img, pts, np.ones((H, W), bool)


def square(y0=40, x0=40, s=30):
    m = np.zeros((H, W), bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_blank_patch_of_wall_is_dropped():
    img, pts, valid = wall()
    assert indistinct({1: square()}, img, pts, valid) == [1]


def test_poster_on_the_wall_is_kept():
    img, pts, valid = wall()
    m = square()
    img[m] = (40, 60, 150)
    assert indistinct({1: m}, img, pts, valid) == []


def test_same_colour_object_standing_out_of_the_wall_is_kept():
    img, pts, valid = wall()
    m = square()
    pts = pts.copy()
    pts[m, 2] -= 0.4                       # a white cabinet in front of a white wall
    assert indistinct({1: m}, img, pts, valid) == []


def test_rug_with_a_border_but_the_floor_colour_is_kept():
    img, pts, valid = wall()
    m = square(s=40)
    inner = square(y0=44, x0=44, s=32)
    img[m & ~inner] = (90, 80, 70)        # a dark border, the field the same as the floor
    assert indistinct({1: m}, img, pts, valid) == []


def test_object_with_little_free_surface_around_it_is_kept():
    img, pts, valid = wall()
    m = square()
    others = np.ones((H, W), np.int32) * 2  # every other pixel belongs to another object
    others[m] = 1
    assert indistinct({1: m}, img, pts, valid, object_pixels=others) == []
