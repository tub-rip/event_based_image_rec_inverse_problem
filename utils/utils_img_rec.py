import numpy as np
import os
from scipy.sparse import dia_matrix, eye
import pylops
import torch
from torch import optim
from utils import utils_image as util
from utils import utils_model
from models.network_unet import UNetRes as net
import torch.nn.functional as F

def computeMatrixA(flow_y, flow_x):
    height, width = flow_y.shape
    flow_y_flat = flow_y.ravel()
    flow_x_flat = flow_x.ravel()
    flow_y_plus = flow_y_flat >= 0.
    flow_x_plus = flow_x_flat >= 0.
    flow_y_minus = np.logical_not(flow_y_plus)
    flow_x_minus = np.logical_not(flow_x_plus)
    flow_y_abs = np.abs(flow_y_flat)
    flow_x_abs = np.abs(flow_x_flat)
    vote_00 = (1.-flow_y_abs) * (1.-flow_x_abs)
    vote_01 = (1.-flow_y_abs) *     flow_x_abs
    vote_10 =     flow_y_abs  * (1.-flow_x_abs)
    vote_11 =     flow_y_abs  *     flow_x_abs
    #t,b,l,r stands for top,bottom,left,right
    mask_tl = np.logical_and(flow_y_minus, flow_x_minus)
    mask_tr = np.logical_and(flow_y_minus, flow_x_plus)
    mask_br = np.logical_and(flow_y_plus , flow_x_plus)
    mask_bl = np.logical_and(flow_y_plus , flow_x_minus)
    #m,t,b,l,r stands for middle,top,bottom,left,right
    vote_tl = mask_tl * vote_11
    vote_tr = mask_tr * vote_11
    vote_br = mask_br * vote_11
    vote_bl = mask_bl * vote_11
    vote_tm = np.logical_or(mask_tl, mask_tr) * vote_10
    vote_bm = np.logical_or(mask_bl, mask_br) * vote_10
    vote_lm = np.logical_or(mask_tl, mask_bl) * vote_01
    vote_rm = np.logical_or(mask_tr, mask_br) * vote_01
    vote_mm = vote_00
    #For Scipy, dia_matrix will throw away the head elements for the upper diagonals.
    #and tail elements for the lower diagonals. In our setting, the head of the vote_tl
    #and the tail of the vote_br should be thrown away. Because the flow of the border 
    # values are zero, so both the head and tails elements of the vote_tl and vote_br 
    # are zeros. So the first and last columns of the matrix A are zeros.
    diagonals_T = np.array([vote_br, vote_bm, vote_bl,
                            vote_rm, vote_mm, vote_lm,
                            vote_tr, vote_tm, vote_tl])
    offsets_T = np.array([-width-1, -width, -width+1,
                            -1,     0,       1,
                           width-1,  width,  width+1])                
    A_T = dia_matrix((diagonals_T, offsets_T), shape=(height*width, height*width))
    A_ = A_T.T
    A_res = eye(height*width) - A_
    
    return A_res

def image_reconstruction(flow_y, flow_x, iwe, reg_type, reg_weight):
    height, width = iwe.shape
    # Build a 9-dignoal matrix
    A_ = computeMatrixA(flow_y, flow_x)
    
    Op = pylops.MatrixMult(A_)
    y_k = iwe.ravel()
    img_init = np.ones(height*width) / 2.0
    Dop = [pylops.FirstDerivative(dims=(height, width), axis=0, edge=False),
           pylops.FirstDerivative(dims=(height, width), axis=1, edge=False)]
    D2op = [pylops.Laplacian(dims=(height, width), edge=True, dtype=np.float64)]
    if reg_type == "l1":
        x_best = pylops.optimization.sparsity.splitbregman(Op, y_k, Dop, niter_outer=20,
                                                           niter_inner=5, mu=1.0,
                                                           epsRL1s=[reg_weight, reg_weight], tol=1e-4,
                                                           tau=1., show=True, x0=img_init,
                                                           **dict(iter_lim=10, damp=1e-4))[0]
    elif reg_type == "l2":
        x_best = pylops.optimization.leastsquares.regularized_inversion(Op, y_k, D2op, epsRs=[reg_weight], 
                                                                        **dict(iter_lim=100, show=True))[0]
    return minmax_norm(x_best.reshape(height, width))

def image_rec_cnn(flow_y, flow_x, iwe, device, model_path, weight1, weight2):
    n_channels = 1
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(os.path.join(model_path, 'drunet_gray.pth')), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    height, width = iwe.shape
    map_y, map_x = np.mgrid[0:height, 0:width]
    map_y = map_y.astype(np.float32)
    map_x = map_x.astype(np.float32)
    warped_y = (map_y + flow_y).reshape(1, height, width)
    warped_x = (map_x + flow_x).reshape(1, height, width)
    warped_y = 2 * warped_y / (height - 1) - 1
    warped_x = 2 * warped_x / (width - 1) - 1
    grid_y = torch.from_numpy(warped_y).float().unsqueeze(0)
    grid_x = torch.from_numpy(warped_x).float().unsqueeze(0)
    grid_pos = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1).to(device)

    # 6.0 is a random number to make the intensity of the final image stay between (0, 1) as much as possible
    img_iwe_ = iwe / (np.amax(np.abs(iwe)) * 6.0)
    y_k = img_iwe_.ravel() 
    A_ = computeMatrixA(flow_y, flow_x)
    img_init = np.ones(height*width) / 2.0
    Op = pylops.MatrixMult(A_)
    Dop = [pylops.FirstDerivative(dims=(height, width), axis=0, edge=False),
           pylops.FirstDerivative(dims=(height, width), axis=1, edge=False)]

    x_best = pylops.optimization.sparsity.splitbregman(Op, y_k, Dop, niter_outer=10, 
                                                       niter_inner=5, mu=1.0, epsRL1s=[7e-2, 7e-2],
                                                       tol=1e-4, tau=1., show=False, x0=img_init,
                                                       **dict(iter_lim=10, damp=1e-4))[0]
    noise_level_img = weight1/255.0
    noise_level_model = noise_level_img
    x8 = True  
    iter_num = 16
    modelSigma1 = 8 
    modelSigma2 = max(0.6, noise_level_model*255.)
    grad_des_iters = 200
    loss = torch.nn.MSELoss()

    img_init = x_best.reshape(1, 1, height, width)
    img_iwe_ = img_iwe_.reshape(1, 1, height, width)
    x = torch.from_numpy(np.ascontiguousarray(img_init)).float().to(device)
    z = x.clone()
    y = torch.from_numpy(np.ascontiguousarray(img_iwe_)).float().to(device)

    rhos_np, sigmas_np = utils_model.get_rho_sigma(sigma=max(0.255/255., noise_level_img) * weight2, iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
    rhos, sigmas = torch.tensor(rhos_np).to(device), torch.tensor(sigmas_np).to(device)

    print("Reconstruct image with CNN denoiser")
    for i in range(iter_num):
        x.requires_grad = True
        optimizer = optim.Adam([x], lr=0.03)
        for it in range(grad_des_iters):
            optimizer.zero_grad()
            warped_x = F.grid_sample(x, grid_pos, mode="bilinear", padding_mode="zeros", align_corners = True)
            output = loss(y, (x - warped_x)) + rhos[i].float() * loss(x, z)
            try:
                output.backward()
            except Exception as e:
                print(e)
            optimizer.step()
        x = x.detach()
        z = x
        if x8:
            z = util.augment_img_tensor4(z, i % 8)
        z = torch.cat((z, sigmas[i].float().repeat(1, 1, z.shape[2], z.shape[3])), dim=1)
        z = utils_model.test_mode(model, z, mode=2, refield=32, min_size=256, modulo=16)
        if x8:
            if i % 8 == 3 or i % 8 == 5:
                z = util.augment_img_tensor4(z, 8 - i % 8)
            else:
                z = util.augment_img_tensor4(z, i % 8)
    x = z
    img_rec = x.data.squeeze().float().cpu().numpy()

    return minmax_norm(img_rec.reshape(height, width))

def minmax_norm(x):
    den = np.percentile(x, 99) - np.percentile(x, 1)
    if den != 0:
        x = (x - np.percentile(x, 1)) / den
    return np.clip(x, 0, 1)


class ImageReconstructor(object):
    def __init__(self, flow_torch):
        """
        flow: (torch.Tensor) [batch_size x 2 x H x W] optical flow, the order of the flow channel is (x, y)
        """
        super().__init__()
        assert flow_torch.ndim == 4
        assert flow_torch.shape[1] == 2
        self.batch_size, _, self.height, self.width = flow_torch.shape
        self.border_mask = np.zeros((self.height, self.width))
        self.border_mask[1:-1, 1:-1] = 1
        self._compute_uni_flow_and_mag(flow_torch)
        self.device = flow_torch.device
        self.flow_torch = flow_torch

    def _compute_uni_flow_and_mag(self, flow_torch):
        def cart2pol(x, y):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return (rho, phi)
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)
        flow_np = flow_torch.cpu().numpy()
        flow_np_x = flow_np[:, 0, :, :].reshape(self.height, self.width)
        flow_np_y = flow_np[:, 1, :, :].reshape(self.height, self.width)
        flow_np_mag, flow_np_ang = cart2pol(flow_np_x, flow_np_y)
        self.flow_np_mag = np.where(flow_np_mag > 0, flow_np_mag, 1.0)
        uni_flow_np_x, uni_flow_np_y = pol2cart(np.ones_like(self.flow_np_mag), flow_np_ang)
        self.uni_flow_np_y = uni_flow_np_y * self.border_mask
        self.uni_flow_np_x = uni_flow_np_x * self.border_mask

    def _check_events(self, events):
        assert events.ndim == 3
        assert events.shape[-1] == 4
        assert len(torch.unique(events[:, :, -1])) == 2
        assert events.shape[0] == self.flow_torch.shape[0]
        # Make sure the events loaded is in range
        assert torch.all(events[:, :, 1] < self.width)
        assert torch.all(events[:, :, 1] > 0)
        assert torch.all(events[:, :, 2] < self.height)
        assert torch.all(events[:, :, 2] > 0)

    def _check_iwe(self, iwe):
        assert iwe.ndim == 4
        assert iwe.shape[1] == 1
        assert iwe.shape[0] == self.flow_torch.shape[0]
    
    def image_rec_from_events_l1(self, events_torch, reg_weight=1e-1):
        """
        events: (torch.Tensor) [batch_size x N x 4] input events (ts, x, y, p), p should either be 0 or 1
        reg_weight: (float) regularization weight
        """
        self._check_events(events_torch)
        iwe = self._compute_iwe(events_torch.to(self.device), self.flow_torch)
        iwe = iwe.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_reconstruction(self.uni_flow_np_y, self.uni_flow_np_x, iwe, "l1", reg_weight)
        return img_rec

    def image_rec_from_events_l2(self, events_torch, reg_weight=3e-1):
        """
        events: (torch.Tensor) [batch_size x N x 4] input events (ts, x, y, p), p should either be 0 or 1
        reg_weight: (float) regularization weight
        """
        self._check_events(events_torch)
        iwe = self._compute_iwe(events_torch.to(self.device), self.flow_torch)
        iwe = iwe.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_reconstruction(self.uni_flow_np_y, self.uni_flow_np_x, iwe, "l2", reg_weight)
        return img_rec
    
    def image_rec_from_events_cnn(self, events_torch, cnn_model_path="denoiser_model", weight1=2.5, weight2=1.3):
        """
        events: (torch.Tensor) [batch_size x N x 4] input events (ts, x, y, p), p should either be 0 or 1
        cnn_model_path: (string) path to the denoiser model
        weight1, weight1: (float) weights to control the regularzation, for details see: https://github.com/cszn/DPIR
        """
        self._check_events(events_torch)
        iwe = self._compute_iwe(events_torch.to(self.device), self.flow_torch)
        iwe = iwe.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_rec_cnn(self.uni_flow_np_y, self.uni_flow_np_x, iwe, self.device, cnn_model_path, weight1, weight2)
        return img_rec

    def image_rec_from_iwe_l1(self, iwe_torch, reg_weight=1e-1):
        """
        iwe_torch: (torch.Tensor) [batch_size x 1 x H x W] image of warped events(with polarity)
        reg_weight: (float) regularization weight
        """
        self._check_iwe(iwe_torch)
        iwe = iwe_torch.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_reconstruction(self.uni_flow_np_y, self.uni_flow_np_x, iwe, "l1", reg_weight)
        return img_rec

    def image_rec_from_iwe_l2(self, iwe_torch, reg_weight=3e-1):
        """
        iwe_torch: (torch.Tensor) [batch_size x 1 x H x W] image of warped events(with polarity)
        reg_weight: (float) regularization weight
        """
        self._check_iwe(iwe_torch)
        iwe = iwe_torch.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_reconstruction(self.uni_flow_np_y, self.uni_flow_np_x, iwe, "l2", reg_weight)
        return img_rec
    
    def image_rec_from_iwe_cnn(self, iwe_torch, cnn_model_path="denoiser_model", weight1=2.5, weight2=1.3):
        """
        iwe_torch: (torch.Tensor) [batch_size x 1 x H x W] image of warped events(with polarity)
        cnn_model_path: (string) path to the denoiser model
        weight1, weight1: (float) weights to control the regularzation, for details see: https://github.com/cszn/DPIR
        """
        self._check_iwe(iwe_torch)
        iwe = iwe_torch.cpu().numpy().reshape(self.height, self.width)
        iwe = iwe * self.border_mask / self.flow_np_mag
        img_rec = image_rec_cnn(self.uni_flow_np_y, self.uni_flow_np_x, iwe, self.device, cnn_model_path, weight1, weight2)
        return img_rec

    def _purge_unfeasible(self, coords):
        """
        Purge unfeasible event locations by setting their interpolation weights to zero.
        """
        mask = torch.ones((coords.shape[0], coords.shape[1], 1)).to(coords.device)
        mask_y = (coords[:, :, 0:1] < 0) + (coords[:, :, 0:1] >= self.height)
        mask_x = (coords[:, :, 1:2] < 0) + (coords[:, :, 1:2] >= self.width)
        mask[mask_y + mask_x] = 0
        return coords * mask, mask
    
    def _interpolate_warped_events(self, warped_events):
        """
        The coordinates (x, y) of the warped events are 'off_grid'(not integer anymore).
        This function spreads the weights of the events to the nearest four cells with bilinear interpolation.
        """
        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        coords = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(warped_events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - coords))

        # purge unfeasible indices
        coords, mask = self._purge_unfeasible(coords)

        # make unfeasible weights zero
        weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

        # prepare indices
        coords[:, :, 0] *= self.width  # torch.view is row-major
        coords = torch.sum(coords, dim=2, keepdim=True)

        return coords, weights

    def _compute_iwe(self, events, flow):

        # Switch from (ts, x, y, p) to (ts, y, x, p)
        events = events[:, :, [0, 2, 1, 3]]
        events[:, :, -1] = events[:, :, -1] * 2 - 1
        # Get x,y of each event
        flow_idx = events[:, :, 1:3].clone()
        # The flow index is x + y * width
        flow_idx[:, :, 0] *= self.width  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for each event
        flow = flow.view(self.batch_size, 2, -1)
        ev_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        ev_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        ev_flowy = ev_flowy.view(ev_flowy.shape[0], ev_flowy.shape[1], 1)
        ev_flowx = ev_flowx.view(ev_flowx.shape[0], ev_flowx.shape[1], 1)
        ev_flow = torch.cat([ev_flowy, ev_flowx], dim=2)

        # warp each event with flow to the timestamp of the last event
        warped_events = events[:, :, 1:3] + (events[:, -1, 0:1] - events[:, :, 0:1]) * ev_flow

        # interpolate each event to the nearest 4 coordinates and scatter the weight
        interpolated_coords, interpolated_weights = self._interpolate_warped_events(warped_events)

        # make negative events have negative weights
        polarities = events[:, :, -1]
        polarities = torch.cat([polarities for i in range(4)], dim=1).unsqueeze(2)
        interpolated_weights *= polarities

        # image of (forward) warped events
        iwe = torch.zeros((self.batch_size, self.height * self.width, 1), dtype=torch.float64).to(warped_events.device)
        iwe = iwe.scatter_add_(1, interpolated_coords.long(), interpolated_weights)
        iwe = iwe.view((self.batch_size, 1, self.height, self.width))

        return iwe