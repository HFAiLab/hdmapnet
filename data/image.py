import torch
import torchvision


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = img.resize(resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran


class Transform:
    def __init__(self, data_conf):
        self.data_conf = data_conf

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / self.data_conf.img_origin_w, fH / self.data_conf.img_origin_h)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def filter_instance_id(self, instance_gt):
        curr_id = 0
        max_id = torch.max(instance_gt)
        while curr_id < max_id:
            if not torch.any(instance_gt == curr_id):
                instance_gt[instance_gt > curr_id] -= 1
                max_id -= 1
            else:
                curr_id += 1
        return instance_gt

    def __call__(self, data_dict):
        imgs = data_dict['imgs']
        data_dict['instance_gt'] = self.filter_instance_id(data_dict['instance_gt'])
        transformed_imgs = []
        post_rots = []
        post_trans = []
        for img in imgs:
            resize, resize_dims = self.sample_augmentation()
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            img = normalize_img(img)
            transformed_imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        data_dict['imgs'] = torch.stack(transformed_imgs)
        data_dict['post_rots'] = torch.stack(post_rots)
        data_dict['post_trans'] = torch.stack(post_trans)
        return data_dict
