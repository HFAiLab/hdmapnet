from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np
import cv2
from easydict import EasyDict

from data.image import denormalize_img


class DataVisualizer:
    def __init__(self, config):
        self.config = config
        self.data_config = self.config.data
        self.cams = self.config.data.cams
        self.input_dim = self.config.data.image_size
        self.vis_output_dir = self.config.runtime.visualize_dir
        if not os.path.exists(self.vis_output_dir):
            os.makedirs(self.vis_output_dir)
        self.visualize_counter = 0

    def visualize_batch(self, data_dict, pred_dict):
        data_dict = EasyDict({k: v.detach().cpu() for k, v in data_dict.items()})
        pred_dict = EasyDict({k: v.detach().cpu() for k, v in pred_dict.items()})
        imgs = data_dict.imgs
        cams = self.cams
        # gt_semantic,    gt_direction,    gt_instance
        # pred_semantic,  pred_direction,  pred_instance
        # img0,           img1,            img2
        # img3,           img4,            img5

        vis_scale_factor = self.config.runtime.vis_scale_factor
        img_h, img_w = self.input_dim
        label_h, label_w = img_w * 0.5, img_w
        plt_w = 3 * img_w * vis_scale_factor
        plt_h = (label_h + label_h + 2 * img_h) * vis_scale_factor
        fig = plt.figure(figsize=(plt_w, plt_h))

        gs = mpl.gridspec.GridSpec(4, 3, height_ratios=(label_h, label_h, img_h, img_h))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        for sample_i in range(imgs.shape[0]):
            plt.clf()
            imname = f'VISUALIZE_batch{self.visualize_counter}_sample{sample_i}.jpg'
            output_file = os.path.join(self.vis_output_dir, imname)

            # visualize gt labels
            color_maps = [cv2.COLORMAP_HSV, cv2.COLORMAP_TURBO, None]
            for label_i, label_type in enumerate(['semantic', 'direction', 'instance']):
                # show gt image
                ax = plt.subplot(gs[0, label_i])
                color_map = color_maps[label_i]
                self.show_img(
                    f'GT_{label_type.upper()}', ax,
                    color_map, data_dict[f'{label_type}_gt'][sample_i],
                    data_dict['semantic_gt'][sample_i],
                    label_type
                )
                # show pred image
                ax = plt.subplot(gs[1, label_i])
                self.show_img(
                    f'{label_type.upper()}', ax,
                    color_map, pred_dict[f'{label_type}'][sample_i],
                    pred_dict['semantic'][sample_i],
                    label_type
                )
            # visualize input img
            for imgi, img in enumerate(imgs[sample_i]):
                    plt.subplot(gs[2 + imgi // 3, imgi % 3])
                    cam_img = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        cam_img = cam_img.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(cam_img)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction', color='r')
            print(f'saving to {output_file}')
            plt.savefig(output_file)
        plt.close()
        self.visualize_counter += 1

    def show_img(self, annotation, ax, color_map, tensor_obj, semantic_tensor, label_type):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if label_type in ['instance']:
            if len(tensor_obj.shape) == 2:
                tensor_obj = tensor_obj.unsqueeze(-1)
            else:
                tensor_obj = np.transpose(tensor_obj, (1, 2, 0))
            show_img = self.visualize_instance(tensor_obj, semantic_tensor)
        else:
            show_img = self.visualize_tensor(tensor_obj, color_map, semantic_tensor)
        plt.imshow(show_img)
        plt.annotate(annotation, (0.01, 0.92), xycoords='axes fraction', color='r')

    @staticmethod
    def visualize_instance(instance_img, semantic_tensor):
        """
            Use the distance of the embedding to visualize
        """
        semantic_mask, _ = DataVisualizer.get_semantic_mask(semantic_tensor)
        instance_img = instance_img / np.linalg.norm(instance_img, axis=2, keepdims=True)

        fg_embedding = np.copy(instance_img)
        fg_embedding[semantic_mask == 1] = 0
        fg_mean = np.mean(fg_embedding, axis=(0, 1))

        instance_dist = np.dot(instance_img, fg_mean)
        min_v, max_v = np.min(instance_dist[semantic_mask == 0]), np.max(instance_dist[semantic_mask == 0])
        instance_dist = (instance_dist - min_v) / (max_v - min_v)
        instance_dist[semantic_mask == 1] = 0

        instance_dist = np.array(255 * instance_dist).astype(np.uint8)
        instances_vis = cv2.applyColorMap(instance_dist, cv2.COLORMAP_RAINBOW)
        instances_vis[semantic_mask] = 255
        return instances_vis

    @staticmethod
    def visualize_tensor(tensor_obj, color_map, semantic_tensor):
        mask, _ = DataVisualizer.get_semantic_mask(semantic_tensor)
        _, viz_img = DataVisualizer.get_semantic_mask(tensor_obj)
        semantic_vis = cv2.applyColorMap(viz_img, color_map)
        semantic_vis[mask] = 255
        return semantic_vis

    @staticmethod
    def get_semantic_mask(tensor_obj):
        cls = tensor_obj.shape[0]
        viz_img = torch.softmax(tensor_obj.float(), dim=0).numpy()
        viz_img = np.argmax(viz_img, axis=0)
        viz_img = np.array(255 * viz_img / cls).astype(np.uint8)
        mask = (viz_img == 0)
        return mask, viz_img

