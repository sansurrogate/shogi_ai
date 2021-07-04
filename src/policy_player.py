import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List

import hydra
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING
from torch.distributions.categorical import Categorical

from lib.features import make_input_features_from_board, make_output_label
from lib.play import Play, Player
from network.policy import PolicyNet


@dataclass
class GreedyChoseActionConfig:
    name: str = "greedy"


@dataclass
class BoltzmannChoseActionConfig:
    name: str = "boltzmann"
    temprature: float = 0.5


defaults = [{"chose_action": "greedy"}]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    modelfile: str = (
        "./outputs/2021-07-03/18-40-58/lightning_logs/"
        "version_0/checkpoints/epoch=2-step=15575.ckpt"
    )
    chose_action: Any = MISSING


def greedy(logits: torch.Tensor):
    return logits.argmax(dim=-1)


def boltzmann(logits: torch.Tensor, temprature):
    m = Categorical(logits=logits / temprature)
    return m.sample()


class PolicyPlayer(Player):
    def __init__(self, modelfile="", chose_action="greedy") -> None:
        super().__init__()
        self.modelfile = modelfile
        self.model: PolicyNet = PolicyNet()
        if chose_action.name == "greedy":
            self.chose_action = greedy
        elif chose_action.name == "boltzmann":
            self.chose_action = partial(
                boltzmann, temprature=chose_action.temprature
            )
        else:
            raise ValueError(
                "chose_action must be either 'greedy' or 'boltzmann'."
            )

    def usi(self) -> str:
        result = (
            "id name policy_player\n"
            f"option name modelfile type string default {self.modelfile}\n"
            "usiok"
        )
        return result

    def setoption(self, option) -> None:
        if option[1] == "modelfile":
            self.modelfile = option[3]

    def isready(self) -> str:
        self.model = PolicyNet.load_from_checkpoint(self.modelfile)
        return "readyok"

    def go(self, board) -> str:
        if board.is_game_over():
            return "bestmove resign"

        features = make_input_features_from_board(board)
        x = torch.tensor(features).unsqueeze(0)
        logits = self.model(x).reshape(-1)
        probabilities = F.log_softmax(logits, dim=-1)

        legal_moves = list(board.legal_moves)
        legal_labels = [make_output_label(m, board.turn) for m in legal_moves]
        legal_logits = logits[legal_labels]
        for move, label in zip(legal_moves, legal_labels):
            logging.info(
                f"{move.usi():5}: "
                f"log_softmax prob.={probabilities[label]:.5g}, "
                f"logits={logits[label]:.5g}"
            )

        selected_index = self.chose_action(legal_logits)
        bestmove = legal_moves[selected_index]
        return f"bestmove {bestmove.usi()}"

    def usinewgame(self) -> None:
        pass

    def quit(self) -> None:
        pass


cs = ConfigStore.instance()
cs.store(group="chose_action", name="greedy", node=GreedyChoseActionConfig)
cs.store(
    group="chose_action", name="boltzmann", node=BoltzmannChoseActionConfig
)
cs.store(name="config", node=Config)


@hydra.main(config_path=None, config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG,
    )
    player = PolicyPlayer(
        modelfile=to_absolute_path(cfg.modelfile),
        chose_action=cfg.chose_action,
    )
    play = Play(player)
    play.start()


if __name__ == "__main__":
    main()
