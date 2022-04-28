
# CLIP-GEN

[简体中文][[English]](README.md)

本项目在萤火二号集群上用 PyTorch 实现了论文 《HDMapNet: A Local Semantic Map Learning and Evaluation Framework》。

**[[Paper](https://arxiv.org/abs/2107.06307)] [[Devkit Page](https://tsinghua-mars-lab.github.io/HDMapNet/)] [[5-min video](https://www.youtube.com/watch?v=AJ-rToTN8y8)]**

![HDMapNet](TODO)

高清地图的构建是自动驾驶的关键问题。这个问题通常涉及收集高质量的点云、融合同一场景的多个点云、标注地图元素以及不断更新地图。然而，这个管道需要大量的人力和资源，这限制了它的可伸缩性。此外，传统的高清地图与厘米级精确定位相结合，这在许多情况下是不可靠的[1]。在本文中，我们认为在线地图学习，基于局部传感器观察动态构建高清地图，是一种比传统的预先注释的高清地图更可扩展的方式，为自动驾驶汽车提供语义和几何优先。同时，介绍了一种在线地图学习方法——HDMapNet。它对周围相机和/或激光雷达的点云的图像特征进行编码，并预测鸟瞰图中的矢量地图元素。我们在nuScenes数据集上对HDMapNet进行了基准测试，结果表明，在所有设置下，HDMapNet的性能都优于基线方法。值得注意的是，我们基于融合的HDMapNet在所有指标上都比现有方法高出50%以上。为了加速未来的研究，我们开发了定制的指标来评估地图学习性能，包括语义级和实例级的指标。通过引入这种方法和度量，我们邀请社区来研究这个新的地图学习问题。我们将发布我们的代码和评估工具包，以促进未来的开发。

## Requirements

- hfai
- torch>=1.8
- nuscenes-devkit

## Preparation

在[config](configs/default.yaml)中设置 `data.dataroot`, `data.version`, `data.batch_size`。

## Training

执行 `python train.py`

## Evaluation

执行 `python eval.py` 

## Demo

下载训练好的模型[HDMapNet_fusion](TODO)，设置[config](configs/default.yaml)中的 `runtime.resume` 为模型路径。

然后执行 `python demo.py`

## Samples

下面是一些可视化效果：

![TODO](TODO)
![TODO](TODO)

## References

- [lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [HDMapNet](https://tsinghua-mars-lab.github.io/HDMapNet)


## Citation

```
@misc{li2021hdmapnet,
    title={HDMapNet: An Online HD Map Construction and Evaluation Framework},
    author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
    year={2021},
    eprint={2107.06307},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

