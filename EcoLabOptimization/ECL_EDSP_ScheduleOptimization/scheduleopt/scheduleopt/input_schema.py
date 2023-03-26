from pydantic import BaseModel, ValidationError, validator

from typing import Union, Dict, List, Any, Optional


class ModelData(BaseModel):
    machine_names: Optional[Dict[str, str]]
    jobs: Dict[str, List[List[List[Any]]]]
    forecast: List[List[Any]]
    batches: Dict[str, Union[float, int]]
    consumption: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]
    initial_amounts: Optional[Dict[str, int]]
    scheduled_shutdown: Optional[List[Dict[str, int]]]
    # due_dates: Optional[List[List[Any]]]

    @validator("jobs")
    def check_jobs(cls, v):
        assert isinstance(v, dict)
        for k, k_job in v.items():
            assert isinstance(k, str)
            assert isinstance(k_job, list)
            for task in k_job:
                assert isinstance(task, list)
                for alt_task in task:
                    assert isinstance(alt_task, list)
                    assert len(alt_task) == 4 or len(alt_task) == 5
                    assert isinstance(alt_task[0], (int, float))
                    assert isinstance(alt_task[1], int)
                    assert isinstance(alt_task[2], str)
                    assert isinstance(alt_task[3], str)
                    if len(alt_task) == 5:
                        assert isinstance(alt_task[4], int)
        return v

    @validator("forecast")
    def check_forecast(cls, v):
        assert isinstance(v, list)
        for item in v:
            assert len(item) == 3
            assert isinstance(item[0], str)
            assert isinstance(item[1], int)
            assert isinstance(item[2], int)
        return v

    @validator("scheduled_shutdown")
    def check_scheduled_shutdown(cls, v):
        assert isinstance(v, list)
        for item in v:
            assert "duration" in item
            assert "minimum_start_time" in item
            assert "maximum_start_time" in item
        return v
