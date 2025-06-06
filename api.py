import requests

base_url = "http://127.0.0.1:8321"

headers_get = {
    "accept": "application/json"
}

headers_post = {
    "Content-Type": "application/json"
}

def train_model():
    url_train_model = f"{base_url}/v1/post-training/preference-optimize"

    train_model_data = {
        "job_uuid": "dpo-training-granite-3.3-2b",
        "model": "ibm-granite/granite-3.3-2b-base",
        "finetuned_model": "granite-3.3-2b-dpo",
        "checkpoint_dir": "./checkpoints",
        "algorithm_config": {
            "type": "dpo",
            "reward_scale": 1.0,
            "reward_clip": 5.0,
            "epsilon": 0.1,
            "gamma": 0.99
        },
        "training_config": {
            "n_epochs": 3,
            "max_steps_per_epoch": 50,
            "learning_rate": 1e-4,
            "warmup_steps": 0,
            "lr_scheduler_type": "constant",
            "data_config": {
                "dataset_id": "test-dpo-dataset-inline-large",
                "batch_size": 2,
                "shuffle": True,
                "data_format": "instruct",
                "train_split_percentage": 0.8
            }
        },
        "hyperparam_search_config": {},
        "logger_config": {}
    }

    response_train_model = requests.post(url_train_model, headers=headers_post, json=train_model_data)
    
    print("Train Model Status:", response_train_model.status_code)
    print("Train Model Response:", response_train_model.json())

    return response_train_model



def get_jobs():
    url_jobs = f"{base_url}/v1/post-training/jobs"
    response_jobs = requests.get(url_jobs, headers=headers_get)
    print("Jobs Status:", response_jobs.status_code)
    print("Jobs Response:", response_jobs.json())

    return response_jobs

def get_job_status(job_uuid):
    url_job_status = f"{base_url}/v1/post-training/job/status?job_uuid={job_uuid}"
    response_job_status = requests.get(url_job_status, headers=headers_get)
    print("Job Status:", response_job_status.status_code)
    print("Job Status Response:", response_job_status.json())

    return response_job_status



def get_job_logs(job_uuid):
    url_job_logs = f"{base_url}/v1/post-training/job/logs?job_uuid={job_uuid}"
    response_job_logs = requests.get(url_job_logs, headers=headers_get)
    print("Job Logs:", response_job_logs.status_code)
    print("Job Logs Response:", response_job_logs.json())

    return response_job_logs

