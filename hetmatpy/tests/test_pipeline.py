from hetmatpy.pipeline import (
    grouper,
)


def test_grouper_equal_chunks():
    iterable = range(10)
    grouped = grouper(iterable, group_size=2)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]


def test_grouper_ragged_chunks():
    iterable = range(7)
    grouped = grouper(iterable, group_size=3)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1, 2),
        (3, 4, 5),
        (6,),
    ]


def test_grouper_one_group():
    iterable = range(7)
    grouped = grouper(iterable, group_size=20)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0, 1, 2, 3, 4, 5, 6),
    ]


def test_grouper_length_1():
    iterable = range(4)
    grouped = grouper(iterable, group_size=1)
    grouped = list(map(tuple, grouped))
    assert grouped == [
        (0,),
        (1,),
        (2,),
        (3,),
    ]
