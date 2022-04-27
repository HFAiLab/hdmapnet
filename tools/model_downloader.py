import os


def download_models():
    src_path = '/ceph-jd/prod/jupyter/bixiao/notebooks/Workspace/Codes/CV/efficient_models/efficientnet-b0-355c32eb.pth'
    target_dir = '/home/bixiao/.cache/torch/hub/checkpoints'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    os.system(
        f"cp {src_path} {target_dir}")
    os.system(f"ls {target_dir}")
