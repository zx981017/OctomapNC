from model import NeuralOcto
from dataset import OctoDataset
from octo_io import read_tree
import torch
from tqdm import tqdm

import wandb


def train(tree,config,data,gt):

    wandb.init(project="octomap", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = data.to(device)
    gt = gt.to(device)

    epoch = config["epoch"]
    learning_rate = config["learning_rate"]
    hidden_dim = config["hidden_dim"]
    hidden_layers = config["hidden_layers"]
    batch_size = config["batch_size"]
    sample_num = config["sample_num"]

    loss = torch.nn.MSELoss()
    best_loss = float('inf')

    model = NeuralOcto(in_features=3, out_features=1, hidden_features=hidden_dim, num_hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    dataset = OctoDataset(data, gt, sample_num)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    model.train()
    for i in tqdm(range(epoch)):
        avg_loss = 0
        for j, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss_value = loss(pred, y)
            loss_value.backward()
            optimizer.step()
            avg_loss += loss_value.item()
        # lr_scheduler.step()
        avg_loss /= len(dataloader)
        wandb.log({"loss": avg_loss})
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
            torch.save(model.state_dict(), "model.pth")

    print("train finished")

    # quantization aware training
    print("Preparing model for QAT...")
    model = best_model
    model.qconfig = torch.quantization.get_default_qconfig(config["qconfig"])
    model = torch.quantization.prepare_qat(model, inplace=True)
    model.train()
    quant_epoch = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    for i in tqdm(range(quant_epoch)):
        avg_loss = 0
        for j, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            loss_value = loss(pred, y)
            loss_value.backward()
            optimizer.step()
            avg_loss += loss_value.item()
        avg_loss /= len(dataloader)
        wandb.log({"loss": avg_loss})
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
    print("Converting model to quantized version...")
    model = torch.quantization.convert(best_model.eval())  # 转换为量化模型
    torch.save(model.state_dict(), "quantized_model.pth")
    print("Quantized model saved as 'quantized_model.pth'")


    return model



if __name__ == "__main__":

    config = {
        "epoch": 200,
        "hidden_dim": 256,
        "hidden_layers": 5,
        "learning_rate": 0.001,
        "batch_size": 64,
        "sample_num": 256,
        "activation": "relu",
        "qconfig": "fbgemm"
    }

    tree = read_tree('fr10cm.ot')
    keys,value = tree.get_node_by_depth()
    depth = 16
    keys = torch.tensor(keys[depth])

    detail_only = False
    if detail_only:
        value = torch.tensor(tree.get_expand_depth_nodes(depth-1)) - torch.tensor(value[depth])
    else:
        value = torch.tensor(value[depth])

    norm = True
    if norm:
        keys = (keys - keys.min()) / (keys.max() - keys.min())
        keys = keys * 2 - 1

    train(tree,config,keys,value)