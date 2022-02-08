# %%
import azureml.core
print("SDK version:", azureml.core.VERSION)
# %%
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
# %%

# %%
import joblib

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

dataset_x, dataset_y = load_diabetes(return_X_y=True)

sk_model = Ridge().fit(dataset_x, dataset_y)

joblib.dump(sk_model, "sklearn_regression_model.pkl")
# %%
from azureml.core.model import Model

model = Model.register(model_path="sklearn_regression_model.pkl",
                       model_name="lyctest",
                       tags={'area': "diabetes", 'type': "regression"},
                       description="Ridge regression model to predict diabetes",
                       workspace=ws)
# %%
import sklearn

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
import  os
environment = Environment("lycLocalDeploy")
dockerfile = r'''
   FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04
   RUN apt-get update -y
   RUN apt-get install -y enchant
   '''
environment.docker.base_image=None   
environment.docker.base_dockerfile=dockerfile
environment.inferencing_stack_version='latest'
# environment.python.conda_dependencies = CondaDependencies(
#             conda_dependencies_file_path= os.path.join(os.getcwd(), 'conda_env.yaml'))
environment.python.conda_dependencies.add_pip_package("inference-schema")
environment.python.conda_dependencies.add_pip_package("joblib")
environment.python.conda_dependencies.add_pip_package("scikit-learn")
environment.python.conda_dependencies.add_pip_package("pyenchant")

print(environment)
# %%
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script="score.py",
                                   environment=environment)
# %%
from azureml.core.webservice import LocalWebservice

# This is optional, if not provided Docker will choose a random unused port.
deployment_config = LocalWebservice.deploy_configuration(port=6789)

local_service = Model.deploy(ws, "test", [model], inference_config, deployment_config)

local_service.wait_for_deployment()
# %%
import json

sample_input = json.dumps({
    'data': dataset_x[0:2].tolist()
})

local_service.run(sample_input)
# %%
local_service.reload()


# %%
from azureml.core.webservice import AksWebservice
s_services = AksWebservice.list(ws)
print(s_services)
# %%
{service.name: service.state for service in s_services}