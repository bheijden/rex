import itertools
from typing import Dict, Tuple

EDGE_INDEX = 9
FACE_INDEX = 6
CANVAS_INDEX = 2


CWHEEL = {
    "white": 10 * ["#ffffff"],
    "black": 10 * ["#000000"],
    "gray": ["#f8f9fa", "#f1f3f5", "#e9ecef", "#dee2e6", "#ced4da", "#adb5bd", "#868e96", "#495057", "#343a40", "#212529"],
    "red": ["#fff5f5", "#ffe3e3", "#ffc9c9", "#ffa8a8", "#ff8787", "#ff6b6b", "#fa5252", "#f03e3e", "#e03131", "#c92a2a"],
    "pink": ["#fff0f6", "#ffdeeb", "#fcc2d7", "#faa2c1", "#f783ac", "#f06595", "#e64980", "#d6336c", "#c2255c", "#a61e4d"],
    "grape": ["#f8f0fc", "#f3d9fa", "#eebefa", "#e599f7", "#da77f2", "#cc5de8", "#be4bdb", "#ae3ec9", "#9c36b5", "#862e9c"],
    "violet": ["#f3f0ff", "#e5dbff", "#d0bfff", "#b197fc", "#9775fa", "#845ef7", "#7950f2", "#7048e8", "#6741d9", "#5f3dc4"],
    "indigo": ["#edf2ff", "#dbe4ff", "#bac8ff", "#91a7ff", "#748ffc", "#5c7cfa", "#4c6ef5", "#4263eb", "#3b5bdb", "#364fc7"],
    "blue": ["#e7f5ff", "#d0ebff", "#a5d8ff", "#74c0fc", "#4dabf7", "#339af0", "#228be6", "#1c7ed6", "#1971c2", "#1864ab"],
    "cyan": ["#e3fafc", "#c5f6fa", "#99e9f2", "#66d9e8", "#3bc9db", "#22b8cf", "#15aabf", "#1098ad", "#0c8599", "#0b7285"],
    "teal": ["#e6fcf5", "#c3fae8", "#96f2d7", "#63e6be", "#38d9a9", "#20c997", "#12b886", "#0ca678", "#099268", "#087f5b"],
    "green": ["#ebfbee", "#d3f9d8", "#b2f2bb", "#8ce99a", "#69db7c", "#51cf66", "#40c057", "#37b24d", "#2f9e44", "#2b8a3e"],
    "lime": ["#f4fce3", "#e9fac8", "#d8f5a2", "#c0eb75", "#a9e34b", "#94d82d", "#82c91e", "#74b816", "#66a80f", "#5c940d"],
    "yellow": ["#fff9db", "#fff3bf", "#ffec99", "#ffe066", "#ffd43b", "#fcc419", "#fab005", "#f59f00", "#f08c00", "#e67700"],
    "orange": ["#fff4e6", "#ffe8cc", "#ffd8a8", "#ffc078", "#ffa94d", "#ff922b", "#fd7e14", "#f76707", "#e8590c", "#d9480f"],
}
COLORS = list(
    [
        "white",
        "black",
        "gray",
        "red",
        "pink",
        "grape",
        "violet",
        "indigo",
        "blue",
        "cyan",
        "teal",
        "green",
        "lime",
        "yellow",
        "orange",
    ]
)
COLORS_CYCLE = itertools.cycle([c for c in COLORS if c not in ["black", "white", "red"]])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ecolor_fn(name: str, index=EDGE_INDEX):
    assert name in CWHEEL, f"Color not found: {name}. Available colors: {list(CWHEEL.keys())}."
    return CWHEEL[name][index]


def fcolor_fn(name: str, index=FACE_INDEX):
    assert name in CWHEEL, f"Color not found: {name}. Available colors: {list(CWHEEL.keys())}."
    return CWHEEL[name][index]


def ccolor(name: str, index=CANVAS_INDEX):
    assert name in CWHEEL, f"Color not found: {name}. Available colors: {list(CWHEEL.keys())}."
    return CWHEEL[name][index]


def cscheme_fn(cscheme: Dict[str, str], edge_index=EDGE_INDEX, face_index=FACE_INDEX) -> Tuple[AttrDict, AttrDict]:
    """Create a color scheme from a dictionary of colors."""
    edge_colors = AttrDict()
    face_colors = AttrDict()
    for name, color in cscheme.items():
        edge_colors[name] = ecolor_fn(color, index=edge_index)
        face_colors[name] = fcolor_fn(color, index=face_index)
    return edge_colors, face_colors


def get_color_cycle(exclude=None):
    exclude = exclude if exclude is not None else ["black", "white", "red"]
    return itertools.cycle([c for c in COLORS if c not in exclude])


# Determine default color scheme
default_cscheme = {
    "computation": "blue",
    "phase": "yellow",
    "advanced": "green",
    "communication": "cyan",
    "sleep": "gray",
    "delay": "red",
    "scheduled": "black",
    "phase_input": "yellow",
    "excluded": "red",  # Used steps
    "pruned": "red",  # Removed steps
    "used": "gray",  # Removed steps
    "rerouted": "orange",  # Rerouted dependency
    "skip": "green",  # Normal dependency
    "normal": "gray",  # Normal dependency
}
ecolor, fcolor = cscheme_fn(default_cscheme)


class _Cwheel:
    def __init__(self, mode):
        assert mode in ["edge", "face"], f"Invalid mode: {mode}."
        self._mode = mode

    def __getitem__(self, item):
        if self._mode == "edge":
            return CWHEEL[item][EDGE_INDEX]
        elif self._mode == "face":
            return CWHEEL[item][FACE_INDEX]


ewheel = _Cwheel("edge")
fwheel = _Cwheel("face")
