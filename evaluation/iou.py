import torch


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def eval_iou(model, val_loader, visualizer=None):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for data_dict in val_loader:
            data_dict = {k: v.cuda() for k, v in data_dict.items()}
            semantic, embedding, direction = model(data_dict)
            semantic_gt = data_dict['semantic_gt'].float()
            if visualizer is not None:
                pred_dict = {
                    'semantic': semantic,
                    'direction': direction,
                    'instance': embedding
                }
                visualizer.visualize_batch(data_dict, pred_dict)
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    return total_intersects / (total_union + 1e-7)


def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)
