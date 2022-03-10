import argparse
import os

from data.dataset import VisualGroundingDataset

from config.config import cfg
dataset = VisualGroundingDataset('sample_train', cfg)
img, text = dataset[1]

# if __name__ == '__main__':

#     # push to remote code here
#     task = None

#     # task = Task.init(project_name=aip_cfg.clearml.project_name,
#     # task_name=aip_cfg.clearml.task_name,
#     # output_uri=os.path.join(aip_cfg.s3.endpoint_url,aip_cfg.s3.model_artifact_path),
#     # reuse_last_task_id=False)
#     # # GIT_SSL_NO_VERIFY=true MUST be set in order for worker to run. If repo is not public, user and password for git has to be included
#     # task.set_base_docker('{} --env GIT_SSL_NO_VERIFY=true --env TRAINS_AGENT_GIT_USER={} --env TRAINS_AGENT_GIT_PASS={}'.format(aip_cfg.docker.base_image,aip_cfg.git.id,aip_cfg.git.key))
#     # task.execute_remotely(queue_name=aip_cfg.clearml.queue,exit_process=True)

#     # #actual code here
#     import experiment
#     from config.config import cfg

#     task.connect(cfg, name='Model Training Parameters')

#     exp = experiment.Experiment(cfg, task)
#     # exp.run_experiment()
#     # exp.create_torchscript_model('class_model_v2.ckpt')
#     # exp.create_torchscript_cpu_model('id_model4.ckpt')
