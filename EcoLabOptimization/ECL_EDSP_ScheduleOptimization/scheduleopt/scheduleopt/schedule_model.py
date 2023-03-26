import math
import collections
from typing import Union, Any, List, Optional, Dict
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from multiprocessing import cpu_count
from importlib.resources import files

from ortools.sat.python import cp_model

import pandas as pd

from .utils import SolutionPrinter

from .input_schema import ModelData
from .schedule_solution import ScheduleSolution


@dataclass
class TaskInterval:
    min_id: str
    job_id: int
    task_id: int
    alt_id: int
    machine_id: int
    start: Any
    duration: Any
    end: Any
    interval: Any
    is_present: Any
    duration_value: Optional[int] = None
    consumption: Optional[Any] = None
    alternates: Optional[List[Any]] = None


@dataclass
class Job:
    job_id: int
    min_id: str
    batch: int
    tasks: List[TaskInterval]
    is_present: Any
    due_date_num: Optional[int] = None
    due_date: Optional[List[int]] = None
    production_jobs: Optional["Job"] = None
    last_consumed: Optional[int] = 0
    total_consumed: Optional[int] = 0


class ScheduleModel:
    def __init__(
        self,
        input_data: Union[dict, ModelData, str, Path],
        cleaning_matrix: Optional[Dict[int, Union[pd.DataFrame, str, Path]]] = None,
    ):
        self._process_cleaning_matrix(cleaning_matrix)
        self._load_input_data(input_data)
        self._initialize_inputs()

    def _process_cleaning_matrix(
        self, cleaning_matrix: Dict[int, Union[pd.DataFrame, str, Path]] = None
    ):
        """Process cleaning matrix into nested diction with structure of
           {CurrentMIN -> {NextMIN -> CleaningProcess}}.

        Args:
            cleaning_matrix (Dict[int, Union[pd.DataFrame, str, Path]], optional): _description_. Defaults to None.
        """
        output_matrix = {}
        if cleaning_matrix is None:
            r358 = files("scheduleopt.data").joinpath("r358-cleaning-matrix.csv")
            r368 = files("scheduleopt.data").joinpath("r368-cleaning-matrix.csv")
            cleaning_matrix = {0: r358, 5: r368}

        for k, v in cleaning_matrix.items():
            if isinstance(v, (str, Path)):
                data = pd.read_csv(v, index_col=0, header=0).to_dict("index")
            else:
                data = v.to_dict("index")
            output_matrix[k] = data
        self._cleaning_matrix = output_matrix

    def _load_input_data(self, input_data: Union[dict, ModelData, str, Path]):
        """Load provided input data to internal variable

        Args:
            input_data (Union[dict, ModelData, str, Path]): Provide model input data.

        Raises:
            ValueError: Raised if schema is not correct.
        """
        if isinstance(input_data, dict):
            self._input_data = ModelData.parse_obj(input_data)
        elif isinstance(input_data, (str, Path)):
            self._input_data = ModelData.parse_file(input_data)
        elif isinstance(input_data, ModelData):
            self._input_data = input_data
        else:
            raise ValueError(
                "Input data must be a valid dictionary, ModelData object, or filepath to input file."
            )

    def _initialize_inputs(self):
        """Generate initial model inputs and reformat as necessary
        """

        # Scale forecasts from hours delivered to minutes
        time_scale_factor = 60  # input of hours
        self._jobs = self._input_data.jobs
        forecast = collections.defaultdict(list) # Dict[MIN, List[Forecast Dates]]
        for item in self._input_data.forecast:
            item[2] *= time_scale_factor
            forecast[item[0]].append(item[1:])
        self._forecasts = forecast

        self._batches = self._input_data.batches

        machine_names = self._input_data.machine_names
        if machine_names is None:
            machine_names = {}
        self._machine_names = machine_names

        consumption = {}
        for min_id, consume in self._input_data.consumption.items():
            for machine_id, task_values in consume.items():
                for task_id, v in task_values.items():
                    consumption[(min_id, int(machine_id), int(task_id))] = {
                        # consumption comes in lbs/min
                        k2: v2
                        for k2, v2 in v.items()
                    }
        self._consumption = consumption

        self._scheduled_shutdown = self._input_data.scheduled_shutdown

        # convert all hours to minutes
        shutdowns = []
        for shutdown in self._scheduled_shutdown:
            shutdowns.append({k: v * time_scale_factor for k, v in shutdown.items()})
        self._scheduled_shutdown = shutdowns

        # Get change over operation times from data files
        changeover_operations = pd.read_csv(
            files("scheduleopt.data").joinpath("cleaning-times.csv")
        )
        self._changeover_operations = {
            k: v for k, v in changeover_operations.to_dict("split")["data"]
        }

        self._initial_amounts = self._input_data.initial_amounts
        if self._initial_amounts is None:
            self._initial_amounts = {}

        self._initial_amounts = self._input_data.initial_amounts

        # Scale job task lengths from hours to minutes
        for min, job_data in self._jobs.items():
            for task in job_data:
                for alt_task in task:
                    alt_task[0] *= time_scale_factor

    def _get_consumption_jobs(self, forecasts):
        required_jobs = {}
        for k, v in forecasts.items():
            required_jobs[k] = math.ceil(v / self._batches.get(k))

        jobs_data = []
        for k, job in self._jobs.items():
            jobs_data += [job] * required_jobs.get(k, 0)

        return jobs_data

    def _get_required_jobs(self, forecasts=None):
        if forecasts is None:
            forecasts = self._forecasts
        required_jobs = {}
        for k, due_dates in forecasts.items():
            required_jobs[k] = math.ceil(
                sum([v[0] for v in due_dates]) / self._batches.get(k)
            )

        jobs_data = []
        for k, job in self._jobs.items():
            jobs_data += [job] * required_jobs.get(k, 0)

        return jobs_data

    def _get_max_possible_jobs(self, timespan):
        # Get full cycle time for each job definition
        def max_jobs(job):
            duration = 0
            for task in job:
                for alt_task in task:
                    duration += alt_task[0]
            return math.floor(timespan / duration)

        job_durations = {k: max_jobs(v) for k, v in self._jobs.items()}

        jobs_data = []
        for k, job in self._jobs.items():
            jobs_data += [job] * job_durations[k]

        return jobs_data

    def _get_max_horizon(self, jobs_data):
        # Computes maximum horizon dynamically as the sum of all durations.
        horizon = 0
        for job in jobs_data:
            for task in job:
                max_task_duration = 0
                for alternative in task:
                    max_task_duration = max(max_task_duration, alternative[0])
                horizon += max_task_duration

        # Add maximum cleaning time span to horizon
        max_cleaning_time = max(self._changeover_operations.values())
        horizon += max_cleaning_time * len(jobs_data)

        if self._scheduled_shutdown is not None:
            for item in self._scheduled_shutdown:
                horizon += item["duration"]
        return int(math.ceil(horizon)) * 4

    def _create_job_intervals(self, model, jobs_data, horizon, job_id_min=0):
        # Named tuple to store information about created variables.
        task_type = collections.namedtuple("task_type", "start end interval min_id")

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        starts = {}
        ends = {}
        presences = {}  # indexed ],  # by (job_id, task_id, alt_id).

        consumption = {}

        jobs = []

        # Iterate through all jobs

        # Track last min job for job hierarchy
        last_min_jobs = collections.defaultdict(lambda: None)

        total_leftover = 0

        job_id = job_id_min
        for job in jobs_data:
            # LMAS Collections
            current_leftover = 0 # Currently available LMAS
            job_consume_total = 0 # Total LMAS consumed in job
            last_consumed = 0
            task_intervals = []
            job_consumptions = [] # LMAS Jobs
            product_jobs = []
            # TODO: Make dict to generalize
            b_job_is_used = model.NewBoolVar(f"job_is_used_j{job_id}")
            previous_end = None  # save previous task end within a job
            task_job_id = job_id
            for task_id, task in enumerate(job):
                # Find min and max task length within alternative tasks
                min_duration = math.ceil(task[0][0])
                max_duration = math.ceil(task[0][0])
                min_id = task[0][3]

                num_alternatives = len(task)
                all_alternatives = range(num_alternatives)
                for alt_id in range(1, num_alternatives):
                    alt_duration = task[alt_id][0]
                    min_duration = min(min_duration, alt_duration)
                    max_duration = max(max_duration, alt_duration)

                # Create primary interval for task using max and min durations.
                suffix_name = "_j%i_t%i" % (task_job_id, task_id)
                start = model.NewIntVar(0, horizon, "start" + suffix_name)
                duration = model.NewIntVar(
                    min_duration, max_duration, "duration" + suffix_name
                )
                end = model.NewIntVar(0, horizon, "end" + suffix_name)
                interval = model.NewOptionalIntervalVar(
                    start, duration, end, b_job_is_used, "interval" + suffix_name
                )

                # Store the start for the solution.
                starts[(task_job_id, task_id)] = start
                ends[(task_job_id, task_id)] = end

                # Add precedence with previous task in the same job.
                if previous_end is not None:
                    model.Add(start == previous_end).OnlyEnforceIf(b_job_is_used)
                previous_end = end

                # Create alternative intervals.
                if num_alternatives > 1:
                    l_presences = []
                    alt_intervals = []
                    for alt_id in all_alternatives:
                        machine_id = task[alt_id][1]
                        machine_name = task[alt_id][2]
                        min_id = task[alt_id][3]

                        alt_suffix = "_j%i_t%i_a%i" % (task_job_id, task_id, alt_id)
                        l_presence = model.NewBoolVar("presence" + alt_suffix)
                        l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                        l_duration = math.ceil(task[alt_id][0])
                        l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                        l_interval = model.NewOptionalIntervalVar(
                            l_start,
                            l_duration,
                            l_end,
                            l_presence,
                            "interval" + alt_suffix,
                        )
                        l_presences.append(l_presence)

                        # Link the master variables with the local ones.
                        model.Add(start == l_start).OnlyEnforceIf(l_presence)
                        model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                        model.Add(end == l_end).OnlyEnforceIf(l_presence)

                        # Add the local interval to the right machine.

                        # Store the presences for the solution.
                        presences[(task_job_id, task_id, alt_id)] = l_presence
                        all_tasks[(task_job_id, task_id, alt_id)] = task_type(
                            start, end, interval, min_id
                        )

                        alt_intervals.append(
                            TaskInterval(
                                min_id,
                                task_job_id,
                                task_id,
                                alt_id,
                                machine_id,
                                l_start,
                                l_duration,
                                l_end,
                                l_interval,
                                l_presence,
                                l_duration,
                            )
                        )

                    # Select exactly one presence variable.
                    model.Add(sum(l_presences) == 1).OnlyEnforceIf(b_job_is_used)
                    model.Add(sum(l_presences) == 0).OnlyEnforceIf(b_job_is_used.Not())

                    task_interval = TaskInterval(
                        min_id,
                        task_job_id,
                        task_id,
                        -1,
                        machine_id,
                        start,
                        duration,
                        end,
                        interval,
                        b_job_is_used,
                        alt_intervals,
                    )
                else:
                    machine_id = task[0][1]
                    machine_name = task[0][2]
                    min_id = task[0][3]
                    presences[(job_id, task_id, 0)] = b_job_is_used
                    all_tasks[(job_id, task_id, 0)] = task_type(
                        start, end, interval, min_id
                    )
                    task_interval = TaskInterval(
                        min_id,
                        task_job_id,
                        task_id,
                        0,
                        machine_id,
                        start,
                        duration,
                        end,
                        interval,
                        b_job_is_used,
                        task[0][0],
                    )
                task_consume_rate = None if len(task[0]) == 4 else task[0][4]

                ## TODO: Does not work with alt reactor tasks right now
                if task_consume_rate:
                    # Currently only support LMAS consumption
                    k = "LMAS"
                    # if current_leftover > consumption, no consumption needed
                    production_batch_size = self._batches.get(k)
                    task_consume_total = task_consume_rate * max_duration
                    job_consume_total += task_consume_total
                    if task_consume_total > current_leftover:
                        job_id += 1

                        # task will require one or more consume product production job
                        production_total_batches = math.ceil(
                            (task_consume_total - current_leftover)
                            / production_batch_size
                        )
                        production_jobs_data = [
                            self._jobs.get(k)
                        ] * production_total_batches
                        production_jobs = self._create_job_intervals(
                            model, production_jobs_data, horizon, job_id
                        )
                        # set outer job id
                        job_id = production_jobs[-1].job_id
                        production_consume_duration = round(
                            production_batch_size / task_consume_rate
                        )

                        product_jobs += production_jobs

                        # first job must finish before current_leftover / rate
                        prev_job_offset = round(current_leftover / task_consume_rate)
                        last_job = None
                        for n, job in enumerate(production_jobs):
                            # Require hierarchy in jobs to limit solver combinations
                            # (i.e., require 1 followed by 2 to avoid solver for different orders )
                            if last_job is not None:
                                model.Add(job.tasks[0].start > last_job.tasks[0].start)

                            # Production job must finish before
                            model.Add(
                                job.tasks[-1].end < (start + prev_job_offset)
                            ).OnlyEnforceIf(job.is_present)
                            # model.Add(job.tasks[-1].end < start)
                            model.Add(
                                job.tasks[-1].end + 12 >= (start + prev_job_offset)
                            ).OnlyEnforceIf(job.is_present)
                            # add new consume duration
                            prev_job_offset += production_consume_duration
                            last_job = job

                        # Replace current_leftover with new leftover production
                        current_leftover += (
                            production_total_batches * production_batch_size
                            - task_consume_total
                        )
                        job_consumptions += production_jobs
                        jobs += production_jobs
                    else:
                        # There is enough leftover to not require a consume product production
                        current_leftover -= task_consume_total
                    last_consumed = self._batches.get("LMAS") - current_leftover

                task_interval.consumption = self._consumption.get(
                    (min_id, machine_id, task_id), None
                )
                task_intervals.append(task_interval)

            # min_id of last job is product
            job = Job(
                task_job_id,
                min_id,
                int(self._batches.get(min_id, 0)),
                task_intervals,
                b_job_is_used,
                production_jobs=product_jobs,
                last_consumed=last_consumed,
                total_consumed=math.ceil(job_consume_total),
            )
            total_leftover += current_leftover

            # require hierarchy in same min_id jobs
            if min_id != "LMAS":
                if last_min_jobs[min_id] is None:
                    last_min_jobs[min_id] = job
                else:
                    model.Add(job.tasks[0].start > last_min_jobs[min_id].tasks[0].start)
                    last_min_jobs[min_id] = job

            jobs.append(job)
            job_id += 1

        return jobs

    def _create_shutdown_jobs(self, model, jobs, horizon):
        job_id = max([job.job_id for job in jobs]) + 1

        if self._scheduled_shutdown is not None:
            for shutdown in self._scheduled_shutdown:
                # Create primary interval for task using max and min durations.
                min_id = "Shutdown"
                machine_id = -1

                suffix_name = f"_shutdown_j{job_id}_t0"
                start = model.NewIntVar(0, horizon, "start" + suffix_name)
                duration = shutdown["duration"]
                end = model.NewIntVar(0, horizon, "end" + suffix_name)
                interval = model.NewIntervalVar(
                    start, duration, end, "interval" + suffix_name
                )
                task_interval = TaskInterval(
                    min_id, job_id, 0, 0, machine_id, start, duration, end, interval, 1
                )
                model.Add(start >= shutdown["minimum_start_time"])
                model.Add(start <= shutdown["maximum_start_time"])

                job = Job(
                    job_id,
                    min_id,
                    int(self._batches.get(min_id, 0)),
                    [task_interval],
                    1,
                )
                job_id += 1
                jobs.append(job)
        return jobs

    def _create_consumption_constraints(self, model: cp_model.CpModel, jobs: List[Job]):
        self.excesses = []

        for n, job in enumerate(jobs):
            total_other_consumed = 0
            if len(job.production_jobs) == 0:
                continue

            for prod_job in job.production_jobs[:-1]:
                # for prod_job in job.production_jobs:
                # If job is used, prod_job must be present for all 
                # but last job
                model.AddImplication(job.is_present, prod_job.is_present)

            last_prod_job = job.production_jobs[-1]
            consumed_amounts = []

            other_prods = []
            for i, other_job in enumerate(jobs):
                if len(other_job.production_jobs) == 0:
                    continue

                if job.min_id == other_job.min_id and i >= n:
                    # for same min_id, jobs are in hierarchial order
                    # so other_job is after current job
                    continue

                # if job is before current job
                job_after_other_job = model.NewBoolVar(
                    f"j{job.job_id}_after_j{other_job.job_id}"
                )

                # Last LMAS production of other job
                last_other_prod_job = other_job.production_jobs[-1]

                # If job_after_other_job is True, other job end is before current job start
                model.Add(job.tasks[0].start >= other_job.tasks[-1].end).OnlyEnforceIf(
                    job_after_other_job
                )

                # If job_after_other_job is False, other job end is after current job start
                model.Add(job.tasks[0].start < other_job.tasks[-1].end).OnlyEnforceIf(
                    job_after_other_job.Not()
                )

                # Boolean variable for last prod job is usable
                last_prod_job_usable = model.NewBoolVar(
                    f"prod_job_less_12_hours_j{other_job.job_id}_used_by_{job.job_id}"
                )

                # If last_prod_job_usable is True, last_other_prod_job end is greather than or equal to
                # current job start - 12
                model.Add(
                    last_other_prod_job.tasks[0].end >= job.tasks[0].start - 12
                ).OnlyEnforceIf(last_prod_job_usable)

                model.Add(
                    last_other_prod_job.tasks[0].end <= job.tasks[0].start
                ).OnlyEnforceIf(last_prod_job_usable)

                last_prod_job_early = model.NewBoolVar(
                    f"prod_job_before_j{other_job.job_id}_used_by_{job.job_id}"
                )
                model.Add(
                    last_other_prod_job.tasks[0].end <= job.tasks[0].start - 12
                ).OnlyEnforceIf(last_prod_job_early)
                last_prod_job_late = model.NewBoolVar(
                    f"prod_job_after_j{other_job.job_id}_used_by_{job.job_id}"
                )
                model.Add(
                    last_other_prod_job.tasks[0].end > job.tasks[0].start
                ).OnlyEnforceIf(last_prod_job_late)

                job_after_and_lmas_usable = model.NewBoolVar(
                    f"j{job.job_id}_using_j{other_job.job_id}_excess"
                )
                model.Add(
                    sum([last_prod_job_usable, job_after_other_job]) == 2
                ).OnlyEnforceIf(job_after_and_lmas_usable)

                model.Add(
                    sum([last_prod_job_early, last_prod_job_late]) == 1
                ).OnlyEnforceIf(last_prod_job_usable.Not())

                consumed = other_job.last_consumed
                j_consumed = model.NewIntVar(
                    -int(self._batches.get("LMAS")),
                    int(self._batches.get("LMAS")),
                    f"consumed_j{job.job_id}_j{other_job.job_id}",
                )
                model.Add(
                    j_consumed
                    == (
                        -int(consumed)
                        + int(self._batches.get("LMAS"))
                        * last_other_prod_job.is_present
                    )
                ).OnlyEnforceIf(job_after_and_lmas_usable)
                model.Add(j_consumed == 0).OnlyEnforceIf(
                    job_after_and_lmas_usable.Not()
                )
                consumed_amounts.append(j_consumed)
                total_other_consumed += other_job.total_consumed

            consumed_amounts.append(self._initial_amounts.get("LMAS", 0))

            max_amount = total_other_consumed
            excess = model.NewIntVar(
                -max_amount, max_amount, f"excess_at_j{job.job_id}"
            )
            model.Add(excess == sum(consumed_amounts))

            model.Add(excess >= int(job.last_consumed)).OnlyEnforceIf(
                last_prod_job.is_present.Not()
            )

            model.Add(excess < int(job.last_consumed)).OnlyEnforceIf(
                last_prod_job.is_present
            )

            self.excesses.append(excess)

    def _require_initial_production_jobs(
        self, model: cp_model.CpModel, jobs: List[Job]
    ):
        processed_mins = set()
        for n, job in enumerate(jobs):
            # Add requirement that very first job must be
            if len(job.production_jobs) == 1 and job.min_id not in processed_mins:
                other_processed_mins = set()
                # only need first min job

                job_before_other_jobs = []
                for i, other_job in enumerate(jobs):
                    if (
                        job.min_id != other_job.min_id
                        and other_job.tasks[0].machine_id == job.tasks[0].machine_id
                        and other_job.min_id not in other_processed_mins
                    ):
                        job_before_other_job = model.NewBoolVar(
                            f"j{job.job_id}_before_j{other_job.job_id}"
                        )
                        model.Add(
                            job.tasks[0].start < other_job.tasks[0].start
                        ).OnlyEnforceIf(job_before_other_job)
                        model.Add(
                            job.tasks[0].start >= other_job.tasks[0].start
                        ).OnlyEnforceIf(job_before_other_job.Not())
                        other_processed_mins.add(other_job.min_id)
                        job_before_other_jobs.append(job_before_other_job)
                job_before_all_jobs = model.NewBoolVar(f"j{job.job_id}_before_all_jobs")
                model.Add(
                    sum(job_before_other_jobs) == len(job_before_other_jobs)
                ).OnlyEnforceIf(job_before_all_jobs)
                model.Add(
                    sum(job_before_other_jobs) != len(job_before_other_jobs)
                ).OnlyEnforceIf(job_before_all_jobs.Not())
                model.Add(job.production_jobs[0].is_present == 1).OnlyEnforceIf(
                    job_before_all_jobs
                )
                processed_mins.add(job.min_id)

    def _create_changeover_intervals_jobs(self, model, horizon, jobs):
        # Disallow overlap of machine specific tasks between different min_ids
        # Set up clean intervals between end task of job A and start task of Job B
        # for each machine_id
        pass

    def _create_changeover_intervals_task(
        self, model: cp_model.CpModel, horizon: int, jobs: List[Job]
    ):
        # Each task can have a different min task after it
        cleaning_matrix = self._cleaning_matrix
        changeover_operations = self._changeover_operations
        # Named tuple to manipulate solution information.
        lasts = []

        # Add cleaning step intervals after machine_id, min_id combinations that are provided
        # 1. min_id is followed directly by cleaning step
        # 2. min_id is followed by another min_id
        job_tasks = []
        for job in jobs:
            changeover_tasks = []
            job_id = job.job_id
            job_is_present = job.is_present

            for n, task in enumerate(job.tasks):
                task_id = task.task_id
                changeover_tasks.append(task)
                machine_id = task.machine_id
                min_id = task.min_id
                end = task.end
                start = task.start

                # For pure consumption products that do not have forecasts
                if min_id == "LMAS":
                    continue
                    alt_suffix = (
                        f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_recharge"
                    )
                    # Cleaning interval creation
                    l_clean_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_clean_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_clean_start,
                        8 * 60,
                        l_clean_end,
                        job_is_present,
                        "interval" + alt_suffix,
                    )
                    model.Add(end == l_clean_start)
                    changeover_tasks.append(
                        TaskInterval(
                            min_id,
                            job_id,
                            -2,
                            -1,
                            machine_id,
                            l_clean_start,
                            8 * 60,
                            l_clean_end,
                            l_interval,
                            job_is_present,
                        )
                    )
                    continue

                # Check if next task is same machine and min_id
                if n != len(job.tasks) - 1:
                    next_task = job.tasks[n + 1]
                    if (
                        next_task.machine_id == machine_id
                        and next_task.min_id == min_id
                    ):
                        # No need to change over since task.end == next_task.start
                        continue

                if machine_id not in cleaning_matrix:
                    continue
                if min_id not in cleaning_matrix[machine_id]:
                    continue

                alt_suffix = (
                    f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_changeover"
                )
                # Cleaning interval creation
                l_clean_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                l_clean_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                l_duration = model.NewIntVar(0, horizon, "duration" + alt_suffix)
                l_interval = model.NewIntervalVar(
                    l_clean_start,
                    l_duration,
                    l_clean_end,
                    "interval" + alt_suffix,
                )
                model.Add(end == l_clean_start)

                l_last = model.NewBoolVar("job_is_last" + alt_suffix)
                lasts.append(l_last)

                l_changeovers = []
                for other_job in jobs:
                    other_job_id = other_job.job_id
                    other_job_is_present = other_job.is_present
                    for m, other_task in enumerate(other_job.tasks):
                        l_changeover = model.NewBoolVar(
                            f"j{job_id}_followed_j{other_job_id}"
                        )
                        other_machine_id = other_task.machine_id
                        other_min_id = min_id
                        other_task_id = other_task.task_id
                        other_start = other_task.start

                        if machine_id != other_machine_id:
                            continue
                        if job_id == other_job_id and task_id == other_task_id:
                            continue

                        l_changeovers.append(l_changeover)
                        model.Add(l_changeover == 0).OnlyEnforceIf(
                            other_job_is_present.Not()
                        )
                        model.Add(l_changeover <= 1).OnlyEnforceIf(other_job_is_present)

                        clean_type = cleaning_matrix[machine_id][min_id].get(
                            other_min_id, "None"
                        )
                        duration = changeover_operations.get(clean_type, 0)
                        model.Add(l_duration >= duration).OnlyEnforceIf(l_changeover)
                        model.Add(l_duration == duration).OnlyEnforceIf(l_last)
                        model.Add(other_start == l_clean_end).OnlyEnforceIf(
                            l_changeover
                        )
                        model.Add(start > other_start).OnlyEnforceIf(l_last)

                model.Add(sum(l_changeovers) + l_last == 0).OnlyEnforceIf(
                    job_is_present.Not()
                )

                model.Add(sum(l_changeovers) + l_last == 1).OnlyEnforceIf(
                    # job_is_present
                )
                model.Add(sum(l_changeovers) == 1).OnlyEnforceIf(l_last.Not())

                changeover_tasks.append(
                    TaskInterval(
                        min_id,
                        job_id,
                        -1,
                        -1,
                        machine_id,
                        l_clean_start,
                        l_duration,
                        l_clean_end,
                        l_interval,
                        job_is_present,
                    )
                )

                # if machine_id not in cleaning_matrix:
                #     continue
                # if min_id not in cleaning_matrix.get(machine_id):
                #     continue

                # for other_job in jobs:
                #     other_job_id = job.job_id
                #     other_job_is_present = job.is_present

                #     for other_task in other_job.tasks:
                #         other_task_id = task.task_id
                #         other_machine_id = task.machine_id
                #         other_min_id = task.min_id
                #         other_end = task.end

                #         if other_job_id == job_id and task_id == other_task_id:
                #             continue

                #         if other_machine_id != machine_id:
                #             continue
                #         if other_min_id not in cleaning_matrix[machine_id][min_id]:
                #             continue

            job_tasks.append(changeover_tasks)

        for job, tasks in zip(jobs, job_tasks):
            job.tasks = tasks
        self.lasts = lasts

    def _create_changeover_intervals(self, model, horizon, jobs):
        changeover = self._changeover
        cleaning_matrix = self._cleaning_matrix
        changeover_operations = self._changeover_operations
        # Named tuple to manipulate solution information.

        # Add cleaning step intervals after machine_id, min_id combinations that are provided
        # 1. min_id is followed directly by cleaning step
        # 2. min_id is followed by another min_id
        durations = []
        same_min_ids = []
        for job in jobs:
            changeover_tasks = []
            job_is_present = job.is_present
            for n, task in enumerate(job.tasks):
                min_id = task.min_id
                job_id = task.job_id
                machine_id = task.machine_id
                task_id = task.task_id
                start = task.start
                end = task.end

                # check for split tasks (i.e., 40 hr reaction split into 5 8 hour reactions) and prevent changeover intervals between them
                if (
                    n != len(job.tasks) - 1
                    and min_id == job.tasks[n + 1].min_id
                    and machine_id == job.tasks[n + 1].machine_id
                ):
                    continue

                # For pure consumption products that do not have forecasts
                if min_id == "LMAS":
                    continue
                    alt_suffix = (
                        f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_recharge"
                    )
                    l_next_is_not_min_id = model.NewBoolVar("min_id_same" + alt_suffix)
                    # Cleaning interval creation
                    l_clean_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_clean_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_clean_start,
                        8 * 60,
                        l_clean_end,
                        job_is_present,
                        "interval" + alt_suffix,
                    )
                    model.Add(end == l_clean_start)
                    changeover_tasks.append(
                        TaskInterval(
                            min_id,
                            job_id,
                            -2,
                            -1,
                            machine_id,
                            l_clean_start,
                            4,
                            l_clean_end,
                            l_interval,
                            job_is_present,
                        )
                    )

                    continue
                # Check if next task is same machine and min_id
                # if n != len(job.tasks) - 1:
                #     next_task = job.tasks[n+1]
                #     if next_task.machine_id == machine_id and next_task.min_id == min_id:
                #         # No need to change over since task.end == next_task.start
                #         continue

                # If no cleaning needed, don't add interval
                if not changeover.get((machine_id, min_id)):
                    continue
                l_changeovers = []

                # Create an interval after each (machine_id, min_id)
                alt_suffix = f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_clean"

                # Bool for tracking if step is last job
                l_last = model.NewBoolVar("laststep" + alt_suffix)

                #
                l_next_is_not_min_id = model.NewBoolVar("min_id_same" + alt_suffix)
                # Cleaning interval creation
                l_clean_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                l_clean_duration = model.NewIntVar(0, horizon, "duration" + alt_suffix)
                durations.append(l_clean_duration)
                l_clean_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                l_interval = model.NewOptionalIntervalVar(
                    l_clean_start,
                    l_clean_duration,
                    l_clean_end,
                    job_is_present,
                    "interval" + alt_suffix,
                )

                # End of task must equal start of cleaning interval
                model.Add(end == l_clean_start)

                # A (machine_id, min_id ) that requires cleaning can be followed by the same (machine_id, min_id) job or
                # a cleaning interval.
                #
                # This for loop enforces

                for job_other in jobs:
                    job_other_is_present = job_other.is_present
                    for task_other in job_other.tasks:
                        min_id_other = task_other.min_id
                        job_id_other = task_other.job_id
                        machine_id_other = task_other.machine_id
                        task_id_other = task_other.task_id
                        start_other = task_other.start
                        end_other = task_other.end
                        job_other_is_present = task_other.is_present
                        if (
                            machine_id_other != machine_id
                            or min_id != min_id_other
                            or (job_id == job_id_other and task_id == task_id_other)
                        ):
                            continue
                        # If task is last in machine_id, l_clean_start must be >= to all other task ends
                        alt_suffix = f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_other{min_id}_to{task_id_other}"

                        # For a cleaning, if next product is same min_id allow duration to float (interval must exist to
                        # ensure next min_id == current min_id) else clean duration is >= to that provided in
                        # cleaning variable
                        l_changeover = model.NewBoolVar("changeover" + alt_suffix)
                        model.Add(start_other == l_clean_end).OnlyEnforceIf(
                            l_changeover
                        )
                        l_changeovers.append(l_changeover)

                    # # Require only one cleaning interval condition is enforced
                model.Add(sum(l_changeovers) == 1).OnlyEnforceIf(
                    l_next_is_not_min_id.Not()
                )
                model.Add(sum(l_changeovers) >= 0).OnlyEnforceIf(l_next_is_not_min_id)
                model.Add(
                    l_clean_duration >= changeover.get((machine_id, min_id), 0)
                ).OnlyEnforceIf(l_next_is_not_min_id)
                # Add l_interval to resources intervals
                same_min_ids.append(l_next_is_not_min_id)
                changeover_tasks.append(
                    TaskInterval(
                        min_id,
                        job_id,
                        -1,
                        -1,
                        machine_id,
                        l_clean_start,
                        l_clean_duration,
                        l_clean_end,
                        l_interval,
                        l_next_is_not_min_id,
                    )
                )
            job.tasks += changeover_tasks

    def _add_no_overlap_condition(self, model: cp_model.CpModel, jobs: List[Job]):
        all_machines = set()
        intervals_per_resources = collections.defaultdict(list)
        production_intervals = []
        for job in jobs:
            for task in job.tasks:
                intervals_per_resources[task.machine_id].append(task)
                all_machines.add(task.machine_id)

        if -1 in all_machines:
            all_machines.remove(-1)
            shutdown_intervals = [task.interval for task in intervals_per_resources[-1]]
            for machine_id in all_machines:
                production_intervals = [
                    task.interval
                    for task in intervals_per_resources[machine_id]
                    if task.task_id >= 0
                ]
                model.AddNoOverlap(production_intervals + shutdown_intervals)

        # Create machines constraints.
        for machine_id in all_machines:
            intervals = [task.interval for task in intervals_per_resources[machine_id]]
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)

        # create consumption constraints between 0 and 5
        for zero_task in intervals_per_resources[0]:
            if zero_task.consumption is None:
                continue
            for k, zero_consumed_val in zero_task.consumption.items():
                for five_task in intervals_per_resources[5]:
                    if (
                        five_task.consumption is None
                        or k not in five_task.consumption
                        or k not in self._max_consumption
                        or (zero_consumed_val + five_task.consumption[k])
                        <= self._max_consumption[k]
                    ):
                        continue

                    model.AddNoOverlap([zero_task.interval, five_task.interval])

    def solve_least_time_schedule(self, max_time_in_seconds=45):
        """Minimal jobshop problem."""

        jobs_data = self._get_required_jobs()

        horizon = self._get_max_horizon(jobs_data)

        # Create the model.
        model = cp_model.CpModel()

        # Creates job intervals and add to the corresponding machine lists.

        jobs = self._create_job_intervals(model, jobs_data, horizon)

        # Check for consumption
        self._create_consumption_constraints(model, jobs)
        self._require_initial_production_jobs(model, jobs)

        # # Force all jobs to be non-consumption products to be present
        jobs = self._create_shutdown_jobs(model, jobs, horizon)
        job_present = [job.is_present for job in jobs if job.min_id in self._forecasts]
        model.Add(sum(job_present) == len(job_present))

        self._create_changeover_intervals_task(model, horizon, jobs)
        job_ends = [job.tasks[-1].end for job in jobs]

        self._add_no_overlap_condition(model, jobs)

        # Makespan objective.
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, job_ends)

        model.Minimize(makespan)

        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        if max_time_in_seconds is not None:
            solver.parameters.max_time_in_seconds = max_time_in_seconds
        solver.parameters.num_search_workers = cpu_count()

        solution_printer = SolutionPrinter()
        # status = solver.Solve(model, solution_printer)

        # # add job hints
        # for job in jobs:
        #     if job.min_id == "LMAS":
        #         continue

        #     for task in job.tasks:
        #         model.AddHint(task.start, solver.Value(task.start))
        #         model.AddHint(task.end, solver.Value(task.end))

        # solver.parameters.max_time_in_seconds = 6000
        status = solver.Solve(model, solution_printer)

        return ScheduleSolution(
            status,
            solver,
            jobs,
            deepcopy(self._machine_names),
            deepcopy(self._cleaning_matrix),
            deepcopy(self._changeover_operations),
            deepcopy(self._input_data),
        )

    def solve_minimize_delivery_miss(self, timespan=1200, max_time_in_seconds=None):
        """Minimal jobshop problem."""

        jobs_data = self._get_required_jobs()

        horizon = self._get_max_horizon(jobs_data)

        # Create the model.
        model = cp_model.CpModel()

        jobs = self._create_job_intervals(model, jobs_data, horizon)

        self._create_consumption_constraints(model, jobs)
        # self._require_initial_production_jobs(model, jobs)

        jobs = self._create_shutdown_jobs(model, jobs, horizon)

        job_present = [job.is_present for job in jobs if job.min_id in self._forecasts]
        model.Add(sum(job_present) == len(job_present))

        # Accumulate due dates
        due_dates = self._forecasts

        for min_id, v in due_dates.items():
            v = sorted(v, key=lambda x: x[1])
            curr_sum = 0
            rows = []
            for row in v:
                row[0] = curr_sum + row[0]
                curr_sum += row[0]
                rows.append(row)
            due_dates[min_id] = rows

        ## For each due date: Sum total product before due date and subtract from due date requirement
        ## set objective to minimize product misses with no benefit for early production

        n_jobs = collections.defaultdict(list)
        for job in jobs:
            n_jobs[job.tasks[-1].min_id].append(job)

        missed_due_dates = []
        due_date_intervals = []
        due_date_num = 0
        due_durations = []
        due_dates_num_days_missed = []
        for min_id, min_due_dates in due_dates.items():
            min_jobs = n_jobs[min_id]
            batch_size = self._batches.get(min_id)
            total_produced = 0
            job_idx = 0
            for due_date in min_due_dates:
                cum_total = due_date[0]
                num_jobs = math.ceil(
                    (cum_total - total_produced) / self._batches.get(min_id)
                )
                total_produced += num_jobs * batch_size

                if num_jobs > 0:
                    job_idx = job_idx + num_jobs - 1
                    last_due_date_job = min_jobs[job_idx]
                    last_due_date_job.due_date = due_date
                    job_is_used = last_due_date_job.is_present
                    last_task = last_due_date_job.tasks[-1]

                    suffix_name = f"j{job_idx}_due{due_date[1]}"
                    start = model.NewIntVar(0, horizon, "start" + suffix_name)
                    duration = model.NewIntVar(
                        -horizon, horizon, "duration" + suffix_name
                    )
                    end = model.NewIntVar(0, horizon, "end" + suffix_name)
                    interval = model.NewOptionalIntervalVar(
                        start, duration, end, job_is_used, "interval" + suffix_name
                    )

                    # if due date is missed, duration > 0

                    num_days_missed = model.NewIntVar(
                        -horizon, horizon, "days_missed" + suffix_name
                    )
                    missed_due_date = model.NewBoolVar("missed_due_date" + suffix_name)
                    model.Add(last_task.end <= due_date[1]).OnlyEnforceIf(
                        missed_due_date.Not()
                    )
                    model.Add(num_days_missed == 0).OnlyEnforceIf(missed_due_date.Not())

                    model.Add(last_task.end > due_date[1]).OnlyEnforceIf(
                        missed_due_date
                    )
                    model.Add(
                        num_days_missed == last_task.end - due_date[1]
                    ).OnlyEnforceIf(missed_due_date)
                    # model.Add(missed_due_date == 0).OnlyEnforceIf(job_is_used.Not())
                    # model.Add(duration > 0).OnlyEnforceIf(missed_due_date)
                    # model.Add(num_days_missed == duration).OnlyEnforceIf(
                    #     missed_due_date
                    # )
                    due_dates_num_days_missed.append(num_days_missed)
                    due_durations.append(duration)
                    missed_due_dates.append(missed_due_date)
            # Populate due dates for jobs for later processing
            last_due_date = None
            for i in range(len(min_jobs) - 1, -1, -1):
                job = min_jobs[i]
                if job.due_date is not None:
                    last_due_date = job.due_date
                    due_date_num += 1
                else:
                    job.due_date = last_due_date
                job.due_date_num = due_date_num

        self._create_changeover_intervals_task(model, horizon, jobs)
        job_ends = [job.tasks[-1].end for job in jobs]
        self._add_no_overlap_condition(model, jobs)

        # Makespan objective.
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, job_ends)
        # model.Minimize(makespan)

        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        if max_time_in_seconds is not None:
            solver.parameters.max_time_in_seconds = max_time_in_seconds
        solver.parameters.num_search_workers = cpu_count()

        all_starts = []
        for job in jobs:
            for task in job.tasks:
                all_starts.append(task.start)

        # Search heuristics
        # model.AddDecisionStrategy(all_starts, cp_model.CHOOSE_RANDOM, cp_model.SELECT_LOWER_HALF)

        solution_printer = SolutionPrinter()
        # status = solver.Solve(model, solution_printer)

        # construction production totals
        all_days_missed = model.NewIntVar(
            -horizon * len(due_dates_num_days_missed),
            horizon * len(due_dates_num_days_missed),
            "TotalProduct",
        )

        solution_printer = SolutionPrinter()
        # status = solver.Solve(model, solution_printer)

        # for job in jobs:
        #     for task in job.tasks:
        #         model.AddHint(task.start, solver.Value(task.start))
        #         model.AddHint(task.end, solver.Value(task.end))

        model.Add(all_days_missed == sum(due_dates_num_days_missed) + makespan)
        model.Minimize(all_days_missed)

        # Creates the solver and solve.
        # solver = cp_model.CpSolver()
        # if max_time_in_seconds is not None:
        #     solver.parameters.max_time_in_seconds = max_time_in_seconds

        status = solver.Solve(model, solution_printer)

        return ScheduleSolution(
            status,
            solver,
            jobs,
            deepcopy(self._machine_names),
            deepcopy(self._cleaning_matrix),
            deepcopy(self._changeover_operations),
            deepcopy(self._input_data),
        )
