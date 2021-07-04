from functools import reduce

import shogi

MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    LEFT,
    RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    # 駒を成らせる
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
] = range(20)

MOVE_DIRECTION_PROMOTED = [
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
]

MOVE_DIRECTION_LABEL_NUM = len(MOVE_DIRECTION) + 7  # 7は持ち駒の種類数


def bb_rotate_180(bb):
    return reduce(
        lambda x, y: x | y,
        [
            shogi.BB_SQUARES[shogi.SQUARES_L90[shogi.SQUARES_L90[pos]]]
            for pos in shogi.SQUARES
            if bb & shogi.BB_SQUARES[pos] > 0
        ],
        0  # initial value
    )
