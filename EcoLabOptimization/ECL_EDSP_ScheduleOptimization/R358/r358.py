# %%
from scheduleopt import ScheduleModel
from pathlib import Path
import pandas as pd
import json
import numpy as np


# with open("input_lmas_ramp.json") as f:
#     inputs = json.load(f)
with open("input_test/input_sample.json") as f:
    inputs = json.load(f)

model = ScheduleModel(inputs)
# %%
sol = model.solve_least_time_schedule(max_time_in_seconds=None)


# %%
def process_overlaps(o):
    values = []
    for val in o:
        print(val)
        sol.solver.Value(val)
        values.append(sol.solver.Value(val))
    values.append(o[1].Name())
    return values


[process_overlaps(overlap) for overlap in sol.jobs[5].tasks[2].overlaps]
[process_overlaps(overlap) for overlap in sol.jobs[5].tasks[10].overlaps]
#%%
[sol.solver.Value(prod_task) for prod_task in sol.jobs[5].tasks[2].prod_jobs]
#%%
jobs_chart = sol.visualize_jobs()
machines_chart = sol.visualize_machines()
jobs_chart
#%%
prod = sol.cumulative_production.copy()
prod.index = prod.index / 60
ax = prod["LMAS"].plot()
fig = ax.get_figure()
ax.set_ylabel("Quantity of LMAS")
ax.set_xlabel("Hours")
# fig.savefig("outputs/lmas_noshutdown_noinitial.png")

# %%


# %%
min_jobs = [job for job in sol.jobs]
display_jobs = []

for job in min_jobs:
    display_jobs.append(
        {
            "min_id": job.min_id,
            "start": sol.solver.Value(job.tasks[0].start),
            "end": sol.solver.Value(job.tasks[-1].end),
            "present": sol.solver.Value(job.is_present),
        }
    )

display_jobs = pd.DataFrame(display_jobs)
display_jobs
# %%
jobs_chart = sol.visualize_jobs()
machines_chart = sol.visualize_machines()
jobs_chart
# %%
machines_chart

# %%
from copy import deepcopy

initial_input = deepcopy(inputs)
initial_input["initial_amounts"] = {"LMAS": 300}
model = ScheduleModel(initial_input, cleaning_matrix)
# %%
sol = model.solve_minimize_delivery_miss(max_time_in_seconds=1200)
# sol = model.solve_least_time_schedule()
# %%
prod = sol.cumulative_production.copy()
prod.index = prod.index / 60
ax = prod["LMAS"].plot()
fig = ax.get_figure()
ax.set_ylabel("Quantity of LMAS")
ax.set_xlabel("Hours")
ax.set_ylim(0, 360)
# fig.savefig("outputs/lmas_noshutdown_noinitial.png")

# %%
min_jobs = [job for job in sol.jobs if job.min_id != "LMAS"]
display_jobs = list(
    zip(
        [job.min_id for job in min_jobs],
        [sol.solver.Value(job.tasks[0].start) for job in min_jobs],
        [sol.solver.Value(job.tasks[-1].end) for job in min_jobs],
        [sol.solver.Value(ex) for ex in model.excesses],
        [job.last_consumed for job in min_jobs],
        [
            sol.solver.Value(job.production_jobs[-1].is_present)
            if job.production_jobs is not None
            else None
            for job in min_jobs
        ],
    )
)
display_jobs
# %%
jobs_chart = sol.visualize_jobs()
machines_chart = sol.visualize_machines()
jobs_chart
# %%
machines_chart
# %%
from copy import deepcopy

shutdown_input = deepcopy(inputs)
shutdown_input["scheduled_shutdown"] = [
    {"duration": 48, "minimum_start_time": 60, "maximum_start_time": 120}
]
model = ScheduleModel(shutdown_input, cleaning_matrix)
# %%
sol = model.solve_minimize_delivery_miss(max_time_in_seconds=1200)
# sol = model.solve_least_time_schedule()
# %%
prod = sol.cumulative_production.copy()
prod.index = prod.index / 60
ax = prod["LMAS"].plot()
fig = ax.get_figure()
ax.set_ylabel("Quantity of LMAS")
ax.set_xlabel("Hours")
ax.set_ylim(0, 360)
fig.savefig("outputs/lmas_noshutdown_noinitial.png")

# %%
min_jobs = [job for job in sol.jobs if job.min_id != "LMAS"]
display_jobs = list(
    zip(
        [job.min_id for job in min_jobs],
        [sol.solver.Value(job.tasks[0].start) for job in min_jobs],
        [sol.solver.Value(job.tasks[-1].end) for job in min_jobs],
        [sol.solver.Value(ex) for ex in model.excesses],
        [job.last_consumed for job in min_jobs],
        [
            sol.solver.Value(job.production_jobs[-1].is_present)
            if job.production_jobs is not None
            else None
            for job in min_jobs
        ],
    )
)
display_jobs
# %%
jobs_chart = sol.visualize_jobs()
machines_chart = sol.visualize_machines()
jobs_chart
# %%
machines_chart


# %%
