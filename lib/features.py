import numpy as np
import shogi

from .common import (
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    LEFT,
    MOVE_DIRECTION,
    MOVE_DIRECTION_PROMOTED,
    RIGHT,
    UP,
    UP2_LEFT,
    UP2_RIGHT,
    UP_LEFT,
    UP_RIGHT,
)


def make_input_features(piece_bb, occupied, pieces_in_hand):
    features = []
    for color in shogi.COLORS:
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9 * 9)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    feature[pos] = 1
            features.append(feature.reshape((9, 9)))
        for piece_type in range(1, 8):
            max_num = shogi.MAX_PIECES_IN_HAND[piece_type]
            number_of_token_in_hand = pieces_in_hand[color].get(piece_type, 0)
            features.extend(
                [np.ones((9, 9)) for _ in range(number_of_token_in_hand)]
                + [
                    np.zeros((9, 9))
                    for _ in range(max_num - number_of_token_in_hand)
                ]
            )
    return np.stack(features, axis=0)


def make_output_label(move, color):
    move_to = move.to_square
    move_from = move.from_square

    if color == shogi.WHITE:
        move_to = shogi.SQUARES_L90[shogi.SQUARES_L90[move_to]]
        if move_from is not None:
            move_from = shogi.SQUARES_L90[shogi.SQUARES_L90[move_from]]

    if move_from is not None:
        to_y, to_x = divmod(move_to, 9)
        from_y, from_x = divmod(move_from, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0 and dir_x == 0:
            move_direction = UP
        elif dir_y == -2 and dir_x == -1:
            move_direction = UP2_LEFT
        elif dir_y == -2 and dir_x == 1:
            move_direction = UP2_RIGHT
        elif dir_y < 0 and dir_x < 0:
            move_direction = UP_LEFT
        elif dir_y < 0 and dir_x > 0:
            move_direction = UP_RIGHT
        elif dir_y == 0 and dir_x < 0:
            move_direction = LEFT
        elif dir_y == 0 and dir_x > 0:
            move_direction = RIGHT
        elif dir_y > 0 and dir_x == 0:
            move_direction = DOWN
        elif dir_y > 0 and dir_x < 0:
            move_direction = DOWN_LEFT
        elif dir_y > 0 and dir_x > 0:
            move_direction = DOWN_RIGHT
        else:
            raise ValueError("invalid direction")

        if move.promotion:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]

    else:
        move_direction = len(MOVE_DIRECTION) + move.drop_piece_type - 1

    move_label = 9 * 9 * move_direction + move_to

    return move_label
