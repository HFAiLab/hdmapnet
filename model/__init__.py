from model.structures.hdmapnet import HDMapNet


def compile_model(data_conf, model_conf):
    if model_conf.method == 'HDMapNet_cam':
        model_conf.lidar = False
        model = HDMapNet(data_conf, model_conf)
    elif model_conf.method == 'HDMapNet_fusion':
        model_conf.lidar = True
        model = HDMapNet(data_conf, model_conf)
    else:
        raise NotImplementedError

    return model
