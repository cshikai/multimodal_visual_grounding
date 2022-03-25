
from clearml import Task, OutputModel

PROJECT_NAME = "multimodal"
MODEL_NAME = "elmo"
OUTPUT_URL = "s3://experiment-logging/multimodal"
LOCAL_MODEL_PATH = '../../models/elmo/'
task = Task.init(project_name=PROJECT_NAME,
                 task_name=MODEL_NAME, output_uri=OUTPUT_URL)

output_model = OutputModel(task=task)
task.update_output_model(model_path=LOCAL_MODEL_PATH)

print('Done')
