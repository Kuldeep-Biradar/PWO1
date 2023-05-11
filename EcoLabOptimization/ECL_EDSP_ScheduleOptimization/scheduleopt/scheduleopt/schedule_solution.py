import collections
import math
from datetime import datetime
from typing import List
from copy import deepcopy
import json

from ortools.sat.python import cp_model

import altair as alt
import pandas as pd
import numpy as np

from .input_schema import ModelData


class ScheduleSolution:
    def __init__(
        self,
        status,
        solver,
        jobs,
        machine_names,
        cleaning_matrix,
        changeover_operations,
        input_data,
        time_scale_factor,
    ):
        self.machine_names = machine_names
        self.input_data = input_data
        self.cleaning_matrix = cleaning_matrix
        self.changeover_operations = changeover_operations
        self.status = status
        self.jobs = jobs
        self.solver = solver
        self._time_scale_factor = time_scale_factor
        self._process_solution()
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self.machine_stats = self._calculate_machine_stats()
            self.job_data = self._aggregate_job_data()
            self._create_time_series()

    def __add__(self, other):
        job_schedule = self.job_schedule.copy().sort_values("Start")
        while job_schedule.iloc[-1]["Expired"] == self.input_data.batches["LMAS"]:
            job_schedule = job_schedule.iloc[:-1]
        # remove lmas jobs that are expired
        other_job_schedule = other.job_schedule.copy().sort_values("Start")
        while (
            other_job_schedule.iloc[-1]["Expired"] == other.input_data.batches["LMAS"]
        ):
            other_job_schedule = other_job_schedule.iloc[:-1]

        max_end_time = max(
            job_schedule.iloc[-1]["End"], other_job_schedule.iloc[-1]["End"]
        )
        other_job_schedule["Start"] += max_end_time
        other_job_schedule["End"] += max_end_time
        combined_job_schedule = pd.concat([job_schedule, other_job_schedule])
        copy = deepcopy(self)
        copy._job_schedule = combined_job_schedule
        copy._time_scale_factor = 1
        copy.machine_stats = copy._calculate_machine_stats()
        copy.job_data = copy._aggregate_job_data()
        copy._create_time_series()

    def _process_solution(
        self,
    ):
        status = self.status
        jobs = self.jobs
        solver = self.solver
        machine_names = self.machine_names

        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            "assigned_task_type",
            "start job index duration min_id, machine_id consumption_rate expiration",
        )

        return_obj = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution_text = ""
            solution_text += "Solution:\n"

            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job in jobs:
                job_id = job.job_id
                # Check if job is used
                if solver.Value(job.is_present) == 0:
                    continue
                for task in job.tasks:
                    task_id = task.task_id

                    # Check if task is used
                    if solver.Value(task.is_present) == 0:
                        continue
                    machine = task.machine_id
                    min_id = task.min_id
                    duration = solver.Value(task.duration)
                    if duration == 0:
                        continue

                    # Used for change over intervals
                    try:
                        if "changeover" in task.end.Name():
                            task_id = "changeover"
                        elif "recharge" in task.end.Name():
                            task_id = "recharge"
                    except:
                        pass
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(task.start),
                            job=job_id,
                            index=task_id,
                            duration=solver.Value(task.duration),
                            min_id=min_id,
                            machine_id=machine,
                            consumption_rate=task.consumption_rate,
                            expiration=solver.Value(task.expired),
                        )
                    )

            # Create per machine output lines.
            output = ""
            for machine, machine_jobs in assigned_jobs.items():
                machine_name = machine_names.get(str(machine), "")

                # Sort by starting time.
                machine_jobs.sort(key=lambda x: x.start)
                sol_line_tasks = "Machine " + str(machine) + ": "
                sol_line = "           "

                for n, assigned_task in enumerate(machine_jobs):
                    activity = f"{assigned_task.min_id} - {machine_name}"
                    operation = activity
                    if isinstance(assigned_task.index, str):
                        operation += "-" + assigned_task.index
                    return_obj.append(
                        {
                            "Machine": machine_name,
                            "MIN": assigned_task.min_id,
                            "JobId": assigned_task.job,
                            "TaskId": assigned_task.index,
                            "Start": assigned_task.start,
                            "End": assigned_task.start + assigned_task.duration,
                            "Activity": activity,
                            "Operation": operation,
                            "MachineId": assigned_task.machine_id,
                            "ConsumptionRate": assigned_task.consumption_rate,
                            "Expired": assigned_task.expiration,
                        }
                    )

                    name = assigned_task.min_id + "_"
                    name += f"{assigned_task.job}_{assigned_task.index}"
                    # Add spaces to output to align columns.
                    sol_line_tasks += "%-15s" % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = "[%i,%i]" % (start, start + duration)
                    # Add spaces to output to align columns.
                    sol_line += "%-15s" % sol_tmp

                sol_line += "\n"
                sol_line_tasks += "\n"
                output += sol_line_tasks
                output += sol_line

            # Finally print the solution found.
            solution_text += f"Optimal Schedule Length: {solver.ObjectiveValue()}\n"
            solution_text += output

            job_schedule = pd.DataFrame(return_obj)

            # Process cleaning operations
            # Changeover intervals are often longer than required due to constraint
            # construction. This operation puts in the required cleaning time rather
            # than the solver's changeover interval.
            cleaning_matrix = self.cleaning_matrix
            changeover_operations = self.changeover_operations

            def fix_intervals(df):
                df = df.sort_values("Start")
                rows = []
                for n, row in df.iterrows():
                    if str(row["TaskId"]) == "changeover":
                        min_id = row["MIN"]
                        if n != 0 and n < df.shape[0] - 1:
                            other_min = df.iloc[n + 1]["MIN"]
                        else:
                            other_min = min_id
                        clean_type = (
                            cleaning_matrix.get(row["MachineId"], {})
                            .get(min_id, {})
                            .get(other_min)
                        )
                        row["TaskId"] = clean_type
                        row["Operation"] = f"{min_id}-{other_min}-{clean_type}"
                        clean_duration = changeover_operations.get(clean_type, 0)
                        row["End"] = row["Start"] + clean_duration
                    rows.append(row)
                return pd.DataFrame(rows)

            self._job_schedule = (
                job_schedule.groupby("MachineId", as_index=False)
                .apply(fix_intervals)
                .reset_index(drop=True)
            )
        else:
            solution_text = (
                f"Solution status: {solver.StatusName()}\nNo solution found!\n"
            )

        # Statistics.
        solution_text += (
            "\nStatistics\n"
            f"  - conflicts: {solver.NumConflicts()}\n"
            f"  - branches : {solver.NumBranches()}\n"
            f"  - wall time: {solver.WallTime()} s\n"
        )

        self.solution_summary = solution_text

    @property
    def job_schedule(self):
        schedule = self._job_schedule.copy()
        schedule["Start"] /= self._time_scale_factor
        schedule["End"] /= self._time_scale_factor
        schedule["ConsumptionRate"] *= (self._time_scale_factor/60)
        return schedule

    def _create_time_series(self):
        job_schedule = self._job_schedule
        job_schedule["ConsumptionRate"] = job_schedule["ConsumptionRate"].fillna(0)
        job_schedule["Expired"] = job_schedule["Expired"].fillna(0)
        production_jobs = job_schedule.loc[
            ~job_schedule["TaskId"].isin(["recharge", "changeover"])
        ]

        production_end_jobs = production_jobs.groupby("JobId").apply(
            lambda x: x.sort_values("End").iloc[-1]
        )
        job_batches = pd.DataFrame(
            [
                {"MIN": k, "Batch": self.input_data.batches.get(k)}
                for k in production_end_jobs["MIN"].unique()
            ]
        )
        production_jobs = (
            production_end_jobs[["MIN", "End"]]
            .reset_index(drop=True)
            .merge(job_batches, on="MIN")
            .rename(columns={"Batch": "Production", "End": "Time"})
        )

        name_to_id = {v: k for k, v in self.machine_names.items()}
        consumption_events = []
        for idx, row in job_schedule.iterrows():
            if isinstance(row["TaskId"], str):
                continue
            consume = row["ConsumptionRate"]
            expiration = row["Expired"]

            if consume > 0:
                for i in range(row["Start"], row["End"]):
                    consumption_events.append(
                        {"MIN": "LMAS", "Consumption": -consume, "Time": i}
                    )
            if expiration > 0:
                consumption_events.append(
                    {"MIN": "LMAS", "Expiration": -expiration, "Time": row["End"]}
                )
        consumption_events = pd.DataFrame(consumption_events)

        production = pd.concat([production_jobs, consumption_events])

        # Check for initial amounts
        initial_amounts = []
        for min_id, amount in self.input_data.initial_amounts.items():
            initial_amounts.append(
                pd.Series({"MIN": min_id, "Time": 0, "Production": amount})
            )

        production = pd.concat([production, *initial_amounts], ignore_index=True)
        if "Expiration" not in production.columns:
            production["Expiration"] = 0

        production = production.fillna(0)
        production["NetProduction"] = (
            production["Production"]
            + production["Consumption"]
            + production["Expiration"]
        )

        production_pivot = (
            pd.pivot_table(
                production,
                values=["NetProduction", "Production", "Consumption", "Expiration"],
                index="Time",
                columns="MIN",
                aggfunc=np.sum,
            )
            .sort_index()
            .fillna(0)
        )
        production_pivot.index = production_pivot.index / self._time_scale_factor
        net_production = production_pivot["NetProduction"]
        production = production_pivot["Production"]
        consumption = production_pivot["Consumption"]
        expiration = production_pivot["Expiration"]

        cumulative_production = net_production.cumsum()
        cumulative_production = cumulative_production.merge(
            pd.Series(np.arange(0, cumulative_production.index.max()), name="index"),
            left_index=True,
            right_index=True,
            how="outer",
        ).fillna(method="ffill")

        self.consumption = consumption
        self.net_production = net_production
        self.expiration = expiration
        self.production = production
        self.cumulative_production = cumulative_production

    def _aggregate_job_data(self):
        solver = self.solver

        job_data = []
        for job in self.jobs:
            finished_time = solver.Value(job.tasks[-1].end)
            if job.due_date:
                delivery_miss = job.due_date[1] < finished_time
                miss_length = None
                if delivery_miss:
                    miss_length = finished_time - job.due_date[1]
                job_data_pts = {
                    "AmountDue": job.due_date[0],
                    "DueDate": job.due_date[1],
                    "DeliveryMiss": delivery_miss,
                    "MissHours": miss_length,
                    "DueDateNum": job.due_date_num,
                }
            else:
                job_data_pts = {}
            job_data.append(
                {
                    "FinishTime": finished_time,
                    "Batch": job.batch,
                    "MinID": job.min_id,
                    **job_data_pts,
                }
            )

        return pd.DataFrame(job_data)

    def _calculate_machine_stats(self):
        stats = []
        for machine, df in self.job_schedule.groupby("Machine"):
            # filter cleaning
            df = df.loc[~df["Operation"].str.contains("clean")]
            start_time = df["Start"].min()
            end_time = df["End"].max()
            durations = df["End"] - df["Start"]

            total_time = end_time - start_time
            running_time = durations.sum()
            stats.append(
                {
                    "Machine": machine,
                    "Start": start_time,
                    "End": end_time,
                    "Total": total_time,
                    "Running": running_time,
                }
            )
        all_data = pd.DataFrame(stats)
        all_data["Start"] = all_data["Start"].min()
        all_data["End"] = all_data["End"].max()
        all_data["Total"] = all_data["End"] - all_data["Start"]
        all_data["Utilization"] = all_data["Running"] / all_data["Total"]

        return all_data

    def to_json(self):
        output = {
            "summary": self.solution_summary,
            "schedule": self.job_schedule.to_csv(),
            "production": self.production.to_csv(),
            "cumulative_production": self.cumulative_production.to_csv(),
        }
        return json.dumps(output)

    def visualize_jobs(self, start_time: datetime = None):
        return self._visualize("jobs", start_time)

    def visualize_machines(self, start_time: datetime = None):
        return self._visualize("machine", start_time)

    def _visualize(self, plot_type, start_time: datetime = None):
        data = self.job_schedule.copy()

        if start_time is None:
            today = datetime.now()
            start_time = datetime(today.year, today.month, 1)

        alt.renderers.enable("jupyterlab")

        data["Start"] = start_time + pd.to_timedelta(data["Start"], unit="hours")
        data["End"] = start_time + pd.to_timedelta(data["End"], unit="hours")

        if plot_type == "jobs":
            return (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x="Start",
                    x2="End",
                    y="Activity",
                    color=alt.Color("Operation", scale=alt.Scale(scheme="dark2")),
                )
                .properties(width=800, height=300)
            )
        elif plot_type == "machine":
            return (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x="Start",
                    x2="End",
                    y="Machine",
                    color=alt.Color("Operation", scale=alt.Scale(scheme="dark2")),
                )
                .properties(width=800, height=300)
            )
