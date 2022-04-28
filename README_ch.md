
# CLIP-GEN

[简体中文][[English]](README.md)

本项目在萤火二号集群上用 PyTorch 实现了论文 《HDMapNet: A Local Semantic Map Learning and Evaluation Framework》。

**[[Paper](https://arxiv.org/abs/2107.06307)] [[Devkit Page](https://tsinghua-mars-lab.github.io/HDMapNet/)] [[5-min video](https://www.youtube.com/watch?v=AJ-rToTN8y8)]**

![HDMapNet](TODO)

高清地图（HD map）构建是自动驾驶的关键问题。这个问题通常涉及收集高质量的点云，融合同一场景的多个点云，标注地图元素，不断更新地图。然而，这条管道需要大量的人力和资源，这限制了它的可扩展性。此外，传统的高清地图加上厘米级的精确定位，这在许多场景中是不可靠的。在本文中，我们认为在线地图学习基于本地传感器观察动态构建高清地图，是一种比传统的预注释高清地图更可扩展的方式来为自动驾驶车辆提供语义和几何先验。同时，我们介绍了一种名为 HDMapNet 的在线地图学习方法。它对来自周围摄像机和/或来自 LiDAR 的点云的图像特征进行编码，并在鸟瞰图中预测矢量化地图元素。我们在 nuScenes 数据集上对 HDMapNet 进行了基准测试，并表明在所有设置中，它的性能都优于基线方法。值得注意的是，我们基于融合的 HDMapNet 在所有指标上都优于现有方法 50% 以上。为了加速未来的研究，我们开发了定制的指标来评估地图学习性能，包括语义级别和实例级别的指标。通过引入这种方法和指标，我们邀请社区研究这个新颖的地图学习问题。我们将发布我们的代码和评估套件以促进未来的开发。我们在 nuScenes 数据集上对 HDMapNet 进行了基准测试，并表明在所有设置中，它的性能都优于基线方法。值得注意的是，我们基于融合的 HDMapNet 在所有指标上都优于现有方法 50% 以上。为了加速未来的研究，我们开发了定制的指标来评估地图学习性能，包括语义级别和实例级别的指标。通过引入这种方法和指标，我们邀请社区研究这个新颖的地图学习问题。我们将发布我们的代码和评估套件以促进未来的开发。我们在 nuScenes 数据集上对 HDMapNet 进行了基准测试，并表明在所有设置中，它的性能都优于基线方法。值得注意的是，我们基于融合的 HDMapNet 在所有指标上都优于现有方法 50% 以上。为了加速未来的研究，我们开发了定制的指标来评估地图学习性能，包括语义级别和实例级别的指标。通过引入这种方法和指标，我们邀请社区研究这个新颖的地图学习问题。我们将发布我们的代码和评估套件以促进未来的开发。包括语义级和实例级。通过引入这种方法和指标，我们邀请社区研究这个新颖的地图学习问题。我们将发布我们的代码和评估套件以促进未来的开发。包括语义级和实例级。通过引入这种方法和指标，我们邀请社区研究这个新颖的地图学习问题。我们将发布我们的代码和评估套件以促进未来的开发。

## Requirements

- hfai
- torch>=1.8
- nuscenes-devkit

## Preparation

在　[config](configs/default.yaml)　中设置 `data.dataroot`, `data.version`, `data.batch_size`。

## Training

执行 `python train.py`

## Evaluation

执行 `python eval.py` 

## Demo

下载训练好的模型 [HDMapNet_fusion](TODO)，设置[config](configs/default.yaml)中的 `runtime.resume` 为模型路径。

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

