from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--workspace_name", type=str, dest="workspace_name", help="workspace name")
parser.add_argument("--resource_group", type=str)
parser.add_argument("--subscription_id", type=str)
parser.add_argument("--filepath", type=str)
args = parser.parse_args()

ws = Workspace(
    subscription_id=args.subscription_id,
    resource_group=args.resource_group,
    workspace_name=args.workspace_name,
)


whl_url = Environment.add_private_pip_wheel(workspace=ws,file_path=args.filepath)
env = Environment(name="endpoint-env")
env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/inference-base-1804:20230201.v1'
conda_dep = CondaDependencies()
conda_dep.add_conda_package('python=3.9')
conda_dep.add_pip_package(whl_url)
env.python.conda_dependencies=conda_dep
env.register(ws)