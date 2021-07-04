import logging
from abc import ABC, abstractmethod

import shogi


class Player(ABC):
    @abstractmethod
    def usi(self) -> str:
        pass

    @abstractmethod
    def setoption(self, option) -> None:
        pass

    @abstractmethod
    def isready(self) -> str:
        pass

    @abstractmethod
    def usinewgame(self) -> None:
        pass

    @abstractmethod
    def go(self, board) -> str:
        pass

    @abstractmethod
    def quit(self) -> None:
        pass


class Play:
    def __init__(self, player: Player) -> None:
        self.board = shogi.Board()
        self.player = player

    def position(self, moves):
        if moves[0] == "startpos":
            self.board.reset()
            for move in moves[2:]:
                self.board.push_usi(move)
        elif moves[0] == "sfen":
            self.board.set_sfen(" ".join(moves[1:]))

        logging.info(self.board.sfen())

    def start(self):
        logging.info("start game. please input command.")
        while True:
            cmd_line = input()
            cmd = cmd_line.split(" ", 1)

            result = None
            if cmd[0] == "position":
                moves = cmd[1].split(" ")
                self.position(moves)
            elif cmd[0] == "usi":
                result = self.player.usi()
            elif cmd[0] == "setoption":
                option = cmd[1].split(" ")
                result = self.player.setoption(option)
            elif cmd[0] == "isready":
                result = self.player.isready()
            elif cmd[0] == "usinewgame":
                result = self.player.usinewgame()
            elif cmd[0] == "go":
                result = self.player.go(self.board)
            elif cmd[0] == "quit":
                self.player.quit()
                break

            if result:
                print(result)
