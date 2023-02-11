import os
import argparse
from utils.utils_viz import save_image
from utils.utils_load import IJRRDataloader
from utils.utils_img_rec import ImageReconstructor


def arg_parser():
    parser = argparse.ArgumentParser(description='Split a big events.txt files to small chunks(1s).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, help='Choose a sequence name from IJRR dataset.')
    parser.add_argument('--start_time', type=float, default=19.5, help="Where we start to load events")
    parser.add_argument('--num_events', type=int, default=30000, help="Number of events to load")
    parser.add_argument('--output_path', type=str, default="results", help="Where to save the output images")
    return parser


def image_reconstruction(args):
    dataset_path = os.path.join("data", args.dataset)
    assert os.path.isdir(dataset_path), "The dataset doesn't exist!!!"
    start_time = args.start_time
    num_events = args.num_events
    output_path = args.output_path
    ijrr_loader = IJRRDataloader(dataset_path)
    events_torch, flow_torch = ijrr_loader.load_events_and_flow(start_time, num_events)
    image_reconstructor = ImageReconstructor(flow_torch)
    img_rec_l1 = image_reconstructor.image_rec_from_events_l1(events_torch, reg_weight=1e-1)
    img_rec_l2 = image_reconstructor.image_rec_from_events_l2(events_torch, reg_weight=3e-1)
    img_rec_denoiser = image_reconstructor.image_rec_from_events_cnn(events_torch, weight1=2.5, weight2=1.3)
    # Save images
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_image(img_rec_l1, output_path, "l1.png")
    save_image(img_rec_l2, output_path, "l2.png")
    save_image(img_rec_denoiser, output_path, "denoiser.png")

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    image_reconstruction(args)