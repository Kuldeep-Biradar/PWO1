import pytest
from scheduleopt.schedule_model import ScheduleModel
from scheduleopt.schedule_solution import ScheduleSolution

def test_schedule_solution_creation(input_data):
    model = ScheduleModel(input_data)
    solution = model.solve_least_time_schedule()
    assert isinstance(solution, ScheduleSolution)