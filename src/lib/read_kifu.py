import copy

import shogi
from shogi import CSA
from tqdm import tqdm

from .common import bb_rotate_180
from .features import make_output_label


def read_kifu(filepath_generator):
    positions = []
    for filepath in tqdm(filepath_generator):
        kifu = CSA.Parser.parse_file(filepath)[0]
        win_color = shogi.BLACK if kifu["win"] == "b" else shogi.WHITE
        board = shogi.Board()
        for move in tqdm(kifu["moves"], leave=False):
            if board.turn == shogi.BLACK:
                piece_bb = copy.deepcopy(board.piece_bb)
                occupied = copy.deepcopy(
                    (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
                )
                pieces_in_hand = copy.deepcopy(
                    (
                        board.pieces_in_hand[shogi.BLACK],
                        board.pieces_in_hand[shogi.WHITE],
                    )
                )
            else:
                piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                occupied = (
                    bb_rotate_180(board.occupied[shogi.WHITE]),
                    bb_rotate_180(board.occupied[shogi.BLACK]),
                )
                pieces_in_hand = copy.deepcopy(
                    (
                        board.pieces_in_hand[shogi.WHITE],
                        board.pieces_in_hand[shogi.BLACK],
                    )
                )

            move_label = make_output_label(
                shogi.Move.from_usi(move), board.turn
            )
            win = 1 if win_color == board.turn else 0
            positions.append(
                (piece_bb, occupied, pieces_in_hand, move_label, win)
            )
            board.push_usi(move)
    return positions
