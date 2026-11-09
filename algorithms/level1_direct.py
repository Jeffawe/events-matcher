"""Level 1 (identity): direct name match.

The sender's name, once normalised, is exactly one person's full name, or a TRUSTED
dictionary alias for exactly one person. Nothing else is attempted here. Anything that
does not match goes to level 2.
"""

from models import Payment

from .dictionary import Dictionary
from .identity import Directory, Identity
from .normalize import normalize


def identify_level1(
    payments: list[Payment], directory: Directory, dictionary: Dictionary
) -> dict[str, Identity]:
    out: dict[str, Identity] = {}
    for pay in payments:
        known, blocked = dictionary.resolve(pay.sender_name)
        people = (known | directory.by_full_name.get(normalize(pay.sender_name), set())) - blocked
        if len(people) == 1:
            (who,) = people
            out[pay.id] = Identity(pay.id, {who: 1.0}, who, "exact", ["full name matches"])
        elif len(people) > 1:
            out[pay.id] = Identity(
                pay.id, {p: 1 / len(people) for p in people}, None, "none",
                ["full name matches more than one person"],
            )
        else:
            out[pay.id] = Identity(pay.id, notes=["no direct name match"])
    return out
