import os
from utils.utils_viz import save_image
from utils.utils_load import load_events_and_flow, load_iwe_and_flow
from utils.utils_img_rec import ImageReconstructor

# Example 1
sample_datasets = ["boxes_rotation", "dynamic_rotation", "poster_rotation", "shapes_rotation"]
output_path = "results"
for dataset in sample_datasets:
    # events_torch: (torch.Tensor) [batch_size x N x 4] input events (ts, x, y, p), p should either be 0 or 1
    # flow_torch: (torch.Tensor) [batch_size x 2 x H x W] optical flow, the order of the flow channel is (x, y)
    events_torch, flow_torch = load_events_and_flow(dataset)
    image_reconstructor = ImageReconstructor(flow_torch)
    img_rec_l1 = image_reconstructor.image_rec_from_events_l1(events_torch, reg_weight=1e-1)
    img_rec_l2 = image_reconstructor.image_rec_from_events_l2(events_torch, reg_weight=3e-1)
    img_rec_denoiser = image_reconstructor.image_rec_from_events_cnn(events_torch, weight1=2.5, weight2=1.3)
    # Save images
    save_path = os.path.join(output_path, dataset)
    save_image(img_rec_l1, save_path, "l1.png")
    save_image(img_rec_l2, save_path, "l2.png")
    save_image(img_rec_denoiser, save_path, "denoiser.png")


# Example 2
dataset = "zurich_city"
output_path = "results"
# iwe_torch: (torch.Tensor) [batch_size x 1 x H x W] image of warped events(with polarity)
# flow_torch: (torch.Tensor) [batch_size x 2 x H x W] optical flow, the order of the flow channel is (x, y)
iwe_torch, flow_torch = load_iwe_and_flow(dataset)
image_reconstructor = ImageReconstructor(flow_torch)
img_rec_l1 = image_reconstructor.image_rec_from_iwe_l1(iwe_torch, reg_weight=1e-1)
img_rec_l2 = image_reconstructor.image_rec_from_iwe_l2(iwe_torch, reg_weight=3e-1)
img_rec_denoiser = image_reconstructor.image_rec_from_iwe_cnn(iwe_torch, weight1=2.5, weight2=1.3)
# Save images
save_path = os.path.join(output_path, dataset)
save_image(img_rec_l1, save_path, "l1.png")
save_image(img_rec_l2, save_path, "l2.png")
save_image(img_rec_denoiser, save_path, "denoiser.png")