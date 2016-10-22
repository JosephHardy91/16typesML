__author__ = 'joe'
from collections import Counter, defaultdict
import sys, os

EXCLUDED_KEYS = ["type", "username"]
TYPE_COLORS = {
    # alpha are blue
    "ILE": "#4286f4",
    "LII": "#1155c1",
    "ESE": "#33b5e0",
    "SEI": "#5da8c1",
    # beta are red
    "SLE": "#cc1414",
    "LSI": "#770d0d",
    "EIE": "#a0553d",
    "IEI": "#d18466",
    # gamma are grey
    "SEE": "#4f4947",
    "ESI": "#706b69",
    "LIE": "#9b8e88",
    "ILI": "#c6beba",
    # delta are green
    "IEE": "#15b528",
    "EII": "#0e6318",
    "LSE": "#43ad4f",
    "SLI": "#669b6c"
}
TYPE_SHAPES = {
    # alpha are blue
    "ILE": "*",
    "LII": "*",
    "ESE": "*",
    "SEI": "*",
    # beta are red
    "SLE": "^",
    "LSI": "^",
    "EIE": "^",
    "IEI": "^",
    # gamma are grey
    "SEE": "o",
    "ESI": "o",
    "LIE": "o",
    "ILI": "o",
    # delta are green
    "IEE": "D",
    "EII": "D",
    "LSE": "D",
    "SLI": "D"
}
