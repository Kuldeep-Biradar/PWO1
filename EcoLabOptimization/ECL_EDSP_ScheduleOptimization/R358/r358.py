# %%
from scheduleopt import ScheduleModel
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
import numpy as np


with open("input_lmas_ramp.json") as f:
    inputs = json.load(f)
# inputs["forecast"] = [["M07A5", 60361, 744]]
# inputs["forecast"] = inputs["forecast4"]
print(inputs["forecast"])
# with open("input_test/input_sample.json") as f:
#     inputs = json.load(f)

model = ScheduleModel(inputs, time_scale_factor=2)
# %%
# sol = model.solve_minimize_delivery_miss(max_time_in_seconds=None, verbose=True)
sol = model.solve_least_time_schedule(max_time_in_seconds=None, verbose=True)
# %%

jobs_chart = sol.visualize_jobs()
machines_chart = sol.visualize_machines()
jobs_chart
#%%

ti = model._time_intervals
d = [
    (
        sol.solver.Value(inte.production),
        sol.solver.Value(inte.consumption),
        sol.solver.Value(inte.state),
        # sol.solver.Value(inte.last_twelve_consumption),
    )
    for inte in ti
]
import pandas as pd

e = [sol.solver.Value(ex) for ex in model._expired]
f = pd.DataFrame(
    d, columns=["Production", "Consumption", "State"]
)
f["Expired"] = e
# f["Inventory"] = f["Production"] + f["Expired"]
ax = f["State"].plot()
f["Expired"].plot()
# f["Production"].plot()
ax.axhline(y=112519, lw=1, color="k")
plt.show()
# %%
# f["RollingConsumption"].plot()

f["RollingProd"] = f["Production"].rolling(window=24).sum().shift(-23)
f["RollingCon"] = f["Consumption"].rolling(window=24).sum().shift(-23)
(f["RollingProd"] - f["RollingCon"] - f["Expired"]).loc[f["Production"] > 0].plot()
# f["Consumption"].plot()
# f["Expired"].plot()
# f["Production"].plot()
plt.show()
#%%
mask = f["Production"] > 0
prev_state = f["State"].iloc[mask.index[mask].values]
prev_state + f["Production"].loc[mask] - f["Expired"].loc[mask] - f["Consumption"].loc[mask]
# %%
expir = f["Expired"].copy()
expir.index /= sol._time_scale_factor
prod = sol.cumulative_production.copy()
prod.index = prod.index / sol._time_scale_factor
ax = prod["LMAS"].plot(label="Production")
expir.plot(ax=ax, label="Expired")
ax.legend()
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
