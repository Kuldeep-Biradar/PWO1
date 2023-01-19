# %% [markdown]
# ### Upgrade/install necessary packages

# %% [markdown]
# ### Adapted Job Shop example from OR-Tools

# %%
"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model

cleaning_time = 100


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print(
            "Solution %i, time = %f s, objective = %i"
            % (self.__solution_count, self.WallTime(), self.ObjectiveValue())
        )
        self.__solution_count += 1


def generate_schedule(jobs_data):
    """Minimal jobshop problem."""

    machines_count = 1 + max(
        alt_task[1] for job in jobs_data for task in job for alt_task in task
    )
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = 0
    for job in jobs_data:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration
    horizon += cleaning_time * len(jobs_data)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval min_id")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration min_id"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    starts = {}
    ends = {}
    min_ends = {}
    min_starts = {}
    job_ends = []
    machine_to_intervals = collections.defaultdict(list)
    intervals_per_resources = collections.defaultdict(list)
    presences = {}  # indexed ],  # by (job_id, task_id, alt_id).

    set_min = False

    for job_id, job in enumerate(jobs_data):
        previous_end = None
        for task_id, task in enumerate(job):
            min_duration = task[0][0]
            max_duration = task[0][0]
            min_id = task[0][3]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = "_j%i_t%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(
                min_duration, max_duration, "duration" + suffix_name
            )
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(
                start, duration, end, "interval" + suffix_name
            )
            
            if not set_min and min_id == "B05Y5" and task_id == 0:
                model.Add(start == 0)
                set_min = True

            # Store the start for the solution.
            starts[(job_id, task_id)] = start
            ends[(job_id, task_id)] = end

            machine_id = task[0][1]
            min_starts[(min_id, machine_id, job_id, task_id)] = start
            min_ends[(min_id, machine_id, job_id, task_id)] = end

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start == previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar("presence" + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                    )
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence
                    all_tasks[(job_id, task_id, alt_id)] = task_type(
                        start, end, interval, min_id
                    )

                # Select exactly one presence variable.
                model.Add(sum(l_presences) == 1)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)
                all_tasks[(job_id, task_id, 0)] = task_type(
                    start, end, interval, min_id
                )
        job_ends.append(previous_end)

    for (min_id, machine_id, job_id, task_id), end in min_ends.items():
        if machine_id != 0:
            continue
        alt_suffix = f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_clean"
        l_changeovers = []
        l_changeover = model.NewBoolVar("changeover" + alt_suffix)
        l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
        l_duration = cleaning_time
        l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
        l_interval = model.NewOptionalIntervalVar(
            l_start, l_duration, l_end, l_changeover, "interval" + alt_suffix
        )
        l_changeovers.append(l_changeover)
        model.Add(end == l_start).OnlyEnforceIf(l_changeover)
        intervals_per_resources[machine_id].append(l_interval)
        for (
            start_min_id,
            start_machine_id,
            start_job_id,
            start_task_id,
        ), start in min_starts.items():
            if (
                start_min_id != min_id
                or start_machine_id != machine_id
                or start_job_id == job_id
            ):
                continue
            alt_suffix = (
                f"_j{job_id}_t{task_id}_m{machine_id}_min{min_id}_ts{start_task_id}"
            )
            l_changeover = model.NewBoolVar("changeover" + alt_suffix)
            model.Add(end == start).OnlyEnforceIf(l_changeover)
            l_changeovers.append(l_changeover)
        model.Add(sum(l_changeovers) >= 1)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Precedences inside a job.
    # for job_id, job in enumerate(jobs_data):
    #     for task_id in range(len(job) - 1):

    #         # Adaptation to allow offsets in start of UF during execution of R368
    #         end_offset = int(all_tasks[job_id, task_id].interval.SizeExpr() * 0.333)
    #         model.Add(
    #             all_tasks[
    #                 job_id,
    #                 task_id + 1,
    #             ].start
    #             >= all_tasks[job_id, task_id].end - end_offset
    #         )

    # Makespan objective.
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)

    # model.AddMaxEquality(
    #     obj_var,
    #     [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    # )
    model.Minimize(makespan)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter()
    status = solver.Solve(model, solution_printer)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                for alt_id, alt_task in enumerate(task):
                    # Check if task is used
                    if solver.Value(presences[(job_id, task_id, alt_id)]) == 0:
                        continue
                    machine = alt_task[1]
                    min_id = alt_task[3]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(
                                all_tasks[(job_id, task_id, alt_id)].start
                            ),
                            job=job_id,
                            index=task_id,
                            duration=alt_task[0],
                            min_id=min_id,
                        )
                    )

        # Create per machine output lines.
        output = ""
        return_dict = {}
        for machine in all_machines:
            machine_name = ""
            if machine == 0:
                machine_name = "Reactor 368"
            if machine == 1:
                machine_name = "UF7"
            if machine == 2:
                machine_name = "UF5"
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                if assigned_task.min_id not in return_dict.keys():
                    return_dict[assigned_task.min_id] = {}
                if machine_name not in return_dict[assigned_task.min_id].keys():
                    return_dict[assigned_task.min_id][machine_name] = {}

                return_dict[assigned_task.min_id][machine_name][assigned_task.job] = [
                    assigned_task.start,
                    assigned_task.start + assigned_task.duration,
                ]
                name = assigned_task.min_id + "_"
                name += "%i_%i" % (assigned_task.job, assigned_task.index)
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
        print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
        print(output)
    else:
        print("No solution found.")

    # Statistics.
    print("\nStatistics")
    print("  - conflicts: %i" % solver.NumConflicts())
    print("  - branches : %i" % solver.NumBranches())
    print("  - wall time: %f s" % solver.WallTime())

    return return_dict


# %%
mo7a5_reqd_batches = 11
bo5y5_reqd_batches = 3

mo7a5_def = [(0, 40, "MO7A5"), (1, 16, "MO7A5")]
b05y5_def = [(0, 10, "BO5Y5"), (1, 8, "BO5Y5")]
jobs_data = []
# Alt R368 setup
jobs = [
    [  # M07A5
        [(40, 0, "R368", "M07A5")],  # task 0
        [(20, 1, "UF7", "M07A5"), (19, 2, "UF5", "M07A5")],  # task 1
    ],
    [  # B05Y5
        [(10, 0, "R368", "B05Y5")],  # task 0
        [(9, 1, "UF7", "B05Y5"), (8, 2, "UF5", "B05Y5")],  # task 1
    ],
]

for i in range(mo7a5_reqd_batches):
    jobs_data.append(jobs[0])
    # jobs_data.append(mo7a5_def)
for i in range(bo5y5_reqd_batches):
    jobs_data.append(jobs[1])
    # jobs_data.append(b05y5_def)

out = generate_schedule(jobs_data)

out

# %%
from datetime import datetime, timedelta

start_time = datetime(2022, 11, 1)

import altair as alt
import datetime as dt
import pandas as pd


alt.renderers.enable("jupyterlab")

data = pd.DataFrame()
from_data = []
to_data = []
activity_data = []


for min_id, mach_vals in out.items():
    for mach_id, job_vals in mach_vals.items():
        for job_id, job_interval in job_vals.items():
            activity = f"{min_id} - {mach_id}"
            begin = start_time + timedelta(hours=job_interval[0])
            end = start_time + timedelta(hours=job_interval[1] - 0.25)
            from_data.append((begin))
            to_data.append((end))
            activity_data.append(activity)

data["Start Time"] = from_data
data["End Time"] = to_data
data["Processing Activity"] = activity_data


alt.Chart(data).mark_bar().encode(
    x="Start Time",
    x2="End Time",
    y="Processing Activity",
    color=alt.Color("Processing Activity", scale=alt.Scale(scheme="dark2")),
).properties(width=800, height=300)

# %%
