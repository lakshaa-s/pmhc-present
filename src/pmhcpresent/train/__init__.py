"""The training dataset and training/evaluation loop for `PresentationNet`."""
from pmhcpresent.train.dataset import PeptideMHCDataset
from pmhcpresent.train.trainer import TrainConfig, evaluate, select_device, train_model
