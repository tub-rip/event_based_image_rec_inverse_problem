# Formulating Event-based Image Reconstruction as a Linear Inverse Problem with Deep Regularization using Optical Flow

The code here is related to the paper: [Formulating Event-based Image Reconstruction as a Linear Inverse Problem with Deep Regularization using Optical Flow](https://arxiv.org/abs/2112.06242).

The paper peoposes a new approach to reconstruct image intensity from events. Assume the optical flow of a short time interval `[t0, t1]` is provided, which can be estimated by any existing 
methods. The events can be warped by the optical flow to a certain timestamp `t(t0<=t<=t1)` and form the image of warped events(IWE). IWE is typically sharp and represents edges(in the direction 
of optical flow) in the scene. The strength(from IWE) and direction(from optical flow) of the edges impose constraints of the image intensity. For example, assume we move the camera vertically 
and upward so that the optical flow is strictly vertical and point downward, the IWE should be full of horizontal edges. At each edge, we know `pixel_intensity_above - pixel_intensity_below =
edge_strength`. These constraints can be formulated as a series of linear equations and form an inverse problem. We reconstruct the image by solving this inverse problem. But of course we need
some regularizations. Please see more details in the paper.

## Instructions to run the code

### Create an environment

Please install [conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), and use the provided `environment.yml` file to install packages:
```
conda env create --file environment.yml
```

### Create folders and download model
Navigate to the root directory of this repository in the terminal and execute:
```
mkdir results && mkdir denoiser_model
```
Then download the CNN denoiser from [here](https://drive.google.com/file/d/1oSsLjPPn6lqtzraFZLZGmwP_5KbPfTES/view?usp=sharing) and put it into the `denoiser_model` folder.
(The model file(drunet_gray.pth) is originally from [here](https://github.com/cszn/DPIR/tree/master/model_zoo))

The `results` folder will be used to store the reconstructed images.

### Run the code
First make sure the conda environment is activated:
```
conda activate img_rec
```
Then execute:
```
python main.py
```

The reconstructed images can be found in `results` folder. 

## Reconstruct image with your own optical flow

The code structure is specially designed to facilitate the process for the user to modifiy the code and reconstruct image if he/she has his/her own optical flow estimation method.

If you want to reconstruct image from events and optical flow:
```
from utils.utils_img_rec import ImageReconstructor
events, flow = load_events_and_flow(dataset)
image_reconstructor = ImageReconstructor(flow)
img_rec_l1 = image_reconstructor.image_rec_from_events_l1(events, reg_weight=1e-1)
img_rec_l2 = image_reconstructor.image_rec_from_events_l2(events, reg_weight=3e-1)
img_rec_denoiser = image_reconstructor.image_rec_from_events_cnn(events, weight1=2.5, weight2=1.3)
```

* In simple words, you just have to provide `flow` to initialize the `ImageReconstructor` and pass the `events` to a specific reconstruction method. You
shall change the regularization weight to obtain the optimal result.
* Please note that the events and flow should be provided in the shape of `[batch_size, N, 4]`, `[batch_size, 2, H, W]` and in `torch.Tensor`.(N means the
number of events, H, W are the height and width of the image resolution). The reason for this convention is the current trend of optical flow estimation
is using deep learning. The flow and events are often converted to `torch.Tensor` and have an extra `batch_size` dimension.

If you want to reconstruct image from IWE and optical flow:
```
from utils.utils_img_rec import ImageReconstructor
iwe, flow = load_iwe_and_flow(dataset)
image_reconstructor = ImageReconstructor(flow)
img_rec_l1 = image_reconstructor.image_rec_from_iwe_l1(iwe, reg_weight=1e-1)
img_rec_l2 = image_reconstructor.image_rec_from_iwe_l2(iwe, reg_weight=3e-1)
img_rec_denoiser = image_reconstructor.image_rec_from_iwe_cnn(iwe, weight1=2.5, weight2=1.3)
```
Similarly, the IWE and flow should be provided in the shape of `[batch_size, 1, H, W]`, `[batch_size, 2, H, W]` and in `torch.Tensor`.

Please see the two examples in the `main.py` for more details.

## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [Deep Plug-and-Play Image Restoration](https://github.com/cszn/DPIR)
- [Back to Event Basics: SSL of Image Reconstruction for Event Cameras](https://github.com/tudelft/ssl_e2vid)
