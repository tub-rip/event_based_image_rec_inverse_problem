import os
import numpy as np
import torch

def load_events_and_flow(dataset):
    dataset_path = os.path.join("sample_dataset", dataset)
    events_np = np.load(os.path.join(dataset_path, "events.npy"))
    flow_np = np.load(os.path.join(dataset_path, "flow.npy"))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    events_torch = torch.from_numpy(events_np).unsqueeze(0).to(device)
    flow_torch = torch.from_numpy(flow_np).to(device)
    return events_torch, flow_torch

def load_iwe_and_flow(dataset):
    dataset_path = os.path.join("sample_dataset", dataset)
    iwe_np = np.load(os.path.join(dataset_path, "iwe.npy"))
    flow_np = np.load(os.path.join(dataset_path, "flow.npy"))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    iwe_torch = torch.from_numpy(iwe_np).to(device)
    flow_torch = torch.from_numpy(flow_np).to(device)
    return iwe_torch, flow_torch