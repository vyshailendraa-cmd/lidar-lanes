"""Submit train_classifier.py as an Azure Machine Learning command job.

One-time setup:
  pip install azure-ai-ml azure-identity
  az login
  Download config.json from your workspace (Azure ML studio > top-right workspace menu)
  and put it in this azure/ folder.

Run (from the repo root):
  python azure/submit_job.py --train data/drive0001.npy --test data/drive0011.npy --test-gt data/drive0011_gt.json

The job runs on cpu-cluster (min 0 nodes, so it costs nothing when idle); metrics, the model and plots appear under
Jobs > lidar-lanes-classifier in Azure ML studio (MLflow is captured automatically).
"""
import argparse
import os

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--test", default=None)
ap.add_argument("--test-gt", default=None, help="hand-labelled lines JSON for the test drive")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
try:
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    cred.get_token("https://management.azure.com/.default")
except Exception:
    cred = InteractiveBrowserCredential()   # opens a browser sign-in if no az login
ml = MLClient.from_config(credential=cred, path=os.path.join(here, "config.json"))

env = Environment(
    name="lidar-lanes-env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    conda_file=os.path.join(here, "conda.yml"),
)

inputs = {"train": Input(type="uri_file", path=args.train)}
cmd = "python train_classifier.py --train ${{inputs.train}} --out outputs/ml"
if args.test:
    inputs["test"] = Input(type="uri_file", path=args.test)
    cmd = "python train_classifier.py --train ${{inputs.train}} --test ${{inputs.test}} --out outputs/ml"
    if args.test_gt:
        inputs["test_gt"] = Input(type="uri_file", path=args.test_gt)
        cmd = cmd.replace(" --out", " --test-gt ${{inputs.test_gt}} --out")

job = command(
    code=os.path.dirname(here),            # repo root: uploads the .py files (data/ and kitti/ are in .amlignore)
    command=cmd,
    inputs=inputs,
    environment=env,
    compute="cpu-cluster",                 # scales to 0 when idle
    display_name="paint-vs-rail-classifier",
    experiment_name="lidar-lanes-classifier",
)
returned = ml.jobs.create_or_update(job)
print("Submitted:", returned.name)
print("Watch it here:", returned.studio_url)
