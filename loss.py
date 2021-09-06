import torch
import torch.nn as nn
#import ChamferDistancePytorch.chamfer2D.dist_chamfer_2D as chamfer2D
from torch.nn.utils.rnn import pad_sequence


def chamfer_distance_without_batch(p1, p2, debug=False):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])

    p1 = p1.repeat(p2.size(1), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(0, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0])

    p2 = p2.repeat(p1.size(0), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=1)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist


def chamfer_distance_with_batch(p1, p2, debug):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    if debug:
        print(p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))
        print(p1[0][0])

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 2)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0][0])

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0][0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=3)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=2)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"
        #self.chamfer_dist = chamfer2D.chamfer_2DDist()

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1


        
        
        #dist1, dist2, idx1, idx2 = self.chamfer_dist(input_points, target_points)
        loss = chamfer_distance_with_batch(input_points, target_points, False)
        #print('chang', loss)
        #loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
