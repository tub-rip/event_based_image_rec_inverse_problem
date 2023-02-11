import os
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Split a big events.txt files to small chunks(1s).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, help='Choose a sequence name from IJRR dataset.')
    return parser


def save_event_chunks(dataset_path):
    save_chunks(dataset_path, "events")


def save_imu_chunks(dataset_path):
    save_chunks(dataset_path, "imu")


def save_chunks(dataset_path, prefix):
    filepath = os.path.join(dataset_path, f'{prefix}.txt')
    save_path = os.path.join(dataset_path, f'{prefix}_chunk/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    with open(filepath, "r") as f:
        g = open(save_path + f"{prefix}_{count}.txt", mode="wt")
        for line in f: 
            components = line.split()
            if len(components) == 0:
                break
            if count <= float(components[0]) and float(components[0]) < count + 1:
                g.write(line)
            else:
                g.close()
                count += 1
                g = open(save_path + f"{prefix}_{count}.txt", mode="wt")


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    dataset_path = os.path.join("data", args.dataset)
    assert os.path.isdir(dataset_path), "The dataset doesn't exist!!!"
    save_event_chunks(dataset_path)
    save_imu_chunks(dataset_path)
