import pytest
from scheduleopt.schedule_model import ScheduleModel


def test_cleaning_matrix_parsing_default_input(input_data):
    """Test that cleaning matrix is successfully initialized with default matrix"""
    model = ScheduleModel(input_data)
    # ModelData schema checks changeover_operations format
    assert isinstance(model._cleaning_matrix, dict)
    assert 0 in model._cleaning_matrix and 5 in model._cleaning_matrix
    assert model._cleaning_matrix[0]["A09B4"]["D07D2"] == "DI Rinse"


def test_cleaning_matrix_parsing_provided_input(input_data, cleaning_input):
    """Test that cleaning matrix is successfully initialized with provided matrix"""
    model = ScheduleModel(input_data, cleaning_input)
    # ModelData schema checks changeover_operations format
    assert isinstance(model._cleaning_matrix, dict)
    assert 0 in model._cleaning_matrix and 5 in model._cleaning_matrix
    assert model._cleaning_matrix[5]["C06F7"]["A03V8"] == "Boilout"


def test_solver_least_time_schedule_cleaning(input_data, cleaning_input):
    model = ScheduleModel(input_data, cleaning_input)
    solution = model.solve_least_time_schedule()
    assert solution.status == 4 or solution.status == 2


def test_solver_max_production_schedule(input_data, cleaning_input):
    model = ScheduleModel(input_data, cleaning_input)
    solution = model.solve_max_production_schedule()
    assert solution.status == 4 or solution.status == 2
    assert solution.solver.ObjectiveValue() > 0


def test_solver_minimize_miss_delivery(input_data, cleaning_input):
    model = ScheduleModel(input_data, cleaning_input)
    solution = model.solve_minimize_delivery_miss()
    assert solution.status == 4 or solution.status == 2
