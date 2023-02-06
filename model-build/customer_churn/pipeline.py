import sagemaker
import boto3
import tarfile
import subprocess
import sys
import json
import string
import os

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker import hyperparameters
from sagemaker.estimator import Estimator
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.model_step import ModelStep

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.functions import (
    JsonGet,
)


sm_client = boto3.client("sagemaker")
sess = sagemaker.Session()
region = boto3.Session().region_name
bucket = sess.default_bucket()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
local_path = "churn.txt"

s3 = boto3.client("s3")
s3.download_file(f"sagemaker-sample-files", "datasets/tabular/synthetic/churn.txt", local_path)

base_uri = f"s3://{bucket}/churn"
input_data_uri = sagemaker.s3.S3Uploader.upload(
    local_path=local_path,
    desired_s3_uri=base_uri,
    
    
)
print(input_data_uri)


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=bucket,
    )
def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket_script=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket_script, key

def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    pipeline_name = "sm-jumpstart-churn-prediction-pipeline",
    model_package_group_name= "ChurnModelPackageGroup",
    base_job_prefix="CustomerChurn",
    default_bucket=bucket
):

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    print(role)

    #Define Pipeline parameters that you can use to parametrize the pipeline. Parameters enable custom pipeline executions and schedules without having to modify the Pipeline definition.

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputData",
        default_value=input_data_uri,
    )

    mse_threshold = ParameterFloat(name="MseThreshold", default_value=6.0)    
    
    
    # processing step for feature engineering#
    framework_version = "0.23-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-churn-process",
        role=role,
        sagemaker_session=sagemaker_session,
    )
    
    step_process = ProcessingStep(
        name="JumpstartDataProcessing",  # choose any name
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation",destination=f"s3://{bucket}/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test",destination=f"s3://{bucket}/output/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocessing.py"),
    )   

    # Estimator Instance count and instance type.
    instance_count = 1
    instance_type = "ml.m5.4xlarge"

    model_id, model_version = "lightgbm-classification-model", "*"
    training_instance_type = "ml.m5.4xlarge"
    
    # Retrieve the docker image
    train_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        model_id=model_id,
        model_version=model_version,
        image_scope="training",
        instance_type=training_instance_type,
    )

    
    # Retrieve the training script
    train_source_uri = script_uris.retrieve(model_id=model_id, model_version=model_version, script_scope="training")

    # Retrieve the pre-trained model tarball to further fine-tune
    train_model_uri = model_uris.retrieve(model_id=model_id, model_version=model_version, model_scope="training")

    # URI of your training dataset
    #training_dataset_s3_path = f"s3:///jumpstart-cache-prod-{region}/training-datasets/tabular_multiclass/"
    training_dataset_s3_path = f"s3://{bucket}/output/train/"

    #output_bucket = sagemaker_session.default_bucket()
    output_prefix = "jumpstart-example-tabular-training"

    s3_output_location = f"s3://{bucket}/{output_prefix}/output"
    
    # Get the default JumpStart hyperparameters
    default_hyperparameters = hyperparameters.retrieve_default(
        model_id=model_id,
        model_version=model_version,
    )
    # [Optional] Override default hyperparameters with custom values
    default_hyperparameters["epochs"] = "1"

    # Unique training job name
    training_job_name = name_from_base(f"built-in-example-{model_id}")

    # Create SageMaker Estimator instance
    ic_estimator = Estimator(
        role=role,
        image_uri=train_image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",
        instance_count=1,
        instance_type=training_instance_type,
        max_run=360000,
        hyperparameters=default_hyperparameters,
        output_path=s3_output_location,
        sagemaker_session=sagemaker_session,
        training=training_dataset_s3_path,
    )
    xgb_input_content_type = None

    training_step = TrainingStep(
        name="JumpStartFineTraining",
        estimator=ic_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
           # "training":training_dataset_s3_path,
        }
    )

    ##Retrieve inference script
    deploy_source_uri = script_uris.retrieve(
        model_id=model_id, model_version=model_version, script_scope="inference"
    )
    bucket_script, key = split_s3_path(deploy_source_uri)
    
    Real_path=os.path.join(BASE_DIR, "sourcedir.tar.gz")
    Inference_dir=os.path.dirname(os.path.realpath(__file__))
    file_name="sourcedir.tar.gz"
    s3.download_file(bucket_script, key, Real_path)
    
    tar = tarfile.open(Real_path)
    tar.extractall(BASE_DIR)
    tar.close()
    reqorg=os.path.join(BASE_DIR, "requirements-final.py")
    reqnew=os.path.join(BASE_DIR, "requirements.txt")
    ##Updating the default requirements file with the libraries we need. This is Work around
    os.remove(reqnew)
    os.rename(reqorg,reqnew)
    inference_instance_type = "ml.m5.4xlarge"
    # Retrieve the inference docker container uri
    deploy_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=inference_instance_type,
    )
    model = Model(
        image_uri=deploy_image_uri,
        entry_point="inference.py",
        source_dir= Inference_dir,
        code_location='s3://' + bucket,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        name="JumpStartRegisterModel",
        role=role,
    )
    
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="s3://{}/evaluation.json".format(bucket),
            content_type="application/json",
       )
    )
    approval_status="Approved"
    
    step_register = RegisterModel(
        name="JumpStartRegisterModel",
        model=model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.4xlarge"],
        transform_instances=["ml.m5.4xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=approval_status,
        model_metrics=model_metrics,
    )    
    pipeline_name = "sm-jumpstart-churn-prediction-pipeline"
    
    # Combine pipeline steps
    #test1
    pipeline_steps = [step_process,training_step,step_register]

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[processing_instance_count,instance_type, instance_count,input_data],
        steps=pipeline_steps,
        sagemaker_session=sagemaker_session
    )
    return pipeline
