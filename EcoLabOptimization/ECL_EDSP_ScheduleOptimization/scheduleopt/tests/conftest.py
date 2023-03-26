from pathlib import Path
import json

import pytest


@pytest.fixture()
def input_data():
    input_path = Path(__file__).parent.joinpath("data/input_lmas_ramp.json")
    with input_path.open() as f:
        input_data = json.load(f)
    return input_data


@pytest.fixture()
def cleaning_input():
    r358_input_path = Path(__file__).parent.joinpath("data/r-358_cleaning_matrix.csv")
    r368_input_path = Path(__file__).parent.joinpath("data/r-368_cleaning_matrix.csv")
    return {0: r358_input_path, 5: r368_input_path}
