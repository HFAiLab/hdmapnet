import torch
from torch import nn


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


class InstanceLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Semantic with a Discriminative Loss Function'
    """

    def __init__(self, loss_config):
        super(InstanceLoss, self).__init__()
        self.delta_var = loss_config.delta_var
        self.delta_dist = loss_config.delta_dist
        self.norm = 'fro'
        self.alpha = loss_config.alpha
        self.beta = loss_config.beta
        self.gamma = 0.001
        self.ignore_index = int(loss_config.ignore_index)

    def _compute_cluster_means(self, input, target):
        embedding_dims = input.size()[1]
        # expand target: NxCxDxHxW -> NxCxExDxHxW
        # NxCx1xDxHxW
        target = target.unsqueeze(2)
        # save target's copy in order to compute the average embeddings later
        target_copy = target.clone()
        shape = list(target.size())
        shape[2] = embedding_dims
        target = target.expand(shape)

        # expand input: NxExDxHxW -> Nx1xExDxHxW
        input = input.unsqueeze(1)

        # sum embeddings in each instance (multiply first via broadcasting) output: NxCxEx1x1x1
        embeddings_per_instance = input * target
        num = torch.sum(embeddings_per_instance, dim=(3, 4, 5), keepdim=True)

        # get number of voxels in each cluster output: NxCx1x1x1x1
        num_voxels_per_instance = torch.sum(target_copy, dim=(3, 4, 5), keepdim=True)

        # compute mean embeddings NxCxEx1x1x1
        mean_embeddings = num / num_voxels_per_instance

        # return mean embeddings and additional tensors needed for further computations
        return mean_embeddings, embeddings_per_instance

    def _compute_variance_term(self, cluster_means, embeddings_per_instance, target):
        # compute the distance to cluster means, result:(NxCxDxHxW)
        embedding_norms = torch.norm(embeddings_per_instance - cluster_means, self.norm, dim=2)

        # get per instance distances (apply instance mask)
        embedding_norms = embedding_norms * target

        # zero out distances less than delta_var and sum to get the variance (NxC)
        embedding_variance = torch.clamp(embedding_norms - self.delta_var, min=0) ** 2
        embedding_variance = torch.sum(embedding_variance, dim=(2, 3, 4))
        # get number of voxels per instance (NxC)
        num_voxels_per_instance = torch.sum(target, dim=(2, 3, 4))
        # normalize the variance term
        C = target.size()[1]
        variance_term = torch.sum(embedding_variance / num_voxels_per_instance, dim=1) / C
        return variance_term

    def _compute_distance_term_naive(self, cluster_means, C):
        if C == 1:
            return 0.

        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)  # N x C x E

        # TODO: N = 1
        cluster_means = cluster_means.squeeze(0)
        simi = torch.mm(cluster_means, cluster_means.T)
        norm = torch.norm(cluster_means, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.T)
        simi_term = simi / norm
        simi_term = torch.sum(simi_term - torch.eye(C).cuda())

        return simi_term / (C * (C - 1))

    def _compute_distance_term(self, cluster_means, C):
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        cluster_means = cluster_means.unsqueeze(1)
        shape = list(cluster_means.size())
        shape[1] = C
        # NxCxCxEx1x1x1
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(0, 2, 1, 3)
        # compute pair-wise distances (NxCxC)
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=3)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        # 1xCxC
        repulsion_dist = repulsion_dist.unsqueeze(0).to(cluster_means.device)
        # zero out distances grater than 2*delta_dist (NxCxC)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2

        # sum all of the hinged pair-wise distances
        hinged_dist = torch.sum(hinged_dist, dim=(1, 2))
        # normalized by the number of paris and return
        return hinged_dist / (C * (C - 1))

    def _compute_regularizer_term(self, cluster_means, C):
        # squeeze space dims
        for _ in range(3):
            cluster_means = cluster_means.squeeze(-1)
        norms = torch.norm(cluster_means, p=self.norm, dim=2)
        assert norms.size()[1] == C
        # return the average norm per batch
        return torch.sum(norms, dim=1).div(C)

    def forward(self, input, target):
        """
        Args:
            input (torch.tensor): embeddings predicted by the network (NxEx1xHxW) (E - embedding dims)
            target (torch.tensor): ground truth instance semantic (Nx1xHxW)

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """

        # TODO:need to modify
        input = input.unsqueeze(2)
        target = target.unsqueeze(1)

        # compare spatial dimensions
        assert input.dim() == 5, f'input: {input.shape}'
        assert target.dim() == 4, f'target: {target.shape}'
        assert input.size()[2:] == target.size()[1:], f'input: {input.size()[2:]} target: {target.size()[1:]}'

        losses = torch.zeros(1).cuda()
        for batch_i in range(input.shape[0]):
            input_i = input[batch_i, :].unsqueeze(0)
            target_i = target[batch_i, :].unsqueeze(0)
            # get number of instances in the batch
            C = int(torch.max(target_i) + 1)

            indices = list(set(range(int(C))) - set([self.ignore_index]))
            keep_indices = torch.LongTensor(indices).cuda()

            if len(indices) == 0:
                continue

            # expand each label as a one-hot vector: N x D x H x W -> N x C x D x H x W
            target_i = expand_as_one_hot(target_i, C)

            # compute mean embeddings and assign embeddings to instances
            cluster_means, embeddings_per_instance = self._compute_cluster_means(input_i, target_i)

            cluster_means = cluster_means[:, keep_indices, :, :, :, :]  # N x C x E x 1 x 1 x 1

            embeddings_per_instance = embeddings_per_instance[:, keep_indices]
            target_i = target_i[:, keep_indices]

            variance_term = self._compute_variance_term(cluster_means, embeddings_per_instance, target_i)
            distance_term = self._compute_distance_term(cluster_means, len(indices))
            regularization_term = self._compute_regularizer_term(cluster_means, len(indices))

            loss = self.alpha * variance_term + self.beta * distance_term + self.gamma * regularization_term
            losses += loss

        return losses / input.shape[0]
