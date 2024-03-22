from enum import IntEnum
from typing import Text
from warnings import warn


class Stance(IntEnum):
    PRO = 1
    CON = 2
    OTHER = 3

    @staticmethod
    def from_label(label: Text) -> 'Stance':
        warn(
            "Infering information from label text is deprecated!",
            DeprecationWarning,
            stacklevel=2,
        )

        if ("P" in label and (label.count("-") % 2 == 0)) or ("K" in label and (label.count("-") % 2 == 1)):
            return Stance.PRO
        elif ("K" in label and (label.count("-") % 2 == 0)) or ("P" in label and (label.count("-") % 2 == 1)):
            return Stance.CON
        else:
            return Stance.OTHER

    @staticmethod
    def from_string(stance_string: Text) -> 'Stance':
        string_to_stance = {
            "pro": Stance.PRO,
            "con": Stance.CON,
        }
        stance_string = stance_string.lower().strip()
        stance = string_to_stance.get(stance_string, Stance.OTHER)
        return stance

    @staticmethod
    def from_agreement_value(agreement: float) -> 'Stance':
        return Stance.PRO if agreement > 0.5 else Stance.CON
