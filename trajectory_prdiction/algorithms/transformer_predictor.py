import torch
import torch.nn as nn
import torch.optim as optim
from trajectory_prdiction.algorithms.nn_transformer import Transformer
from datetime import datetime
import os
import json

class TransformerPredictor:
    def __init__(self,
        input_dim: int = 3,
        embed_dim: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        L_in: int = 10,
        L_out: int = 5,
        device: str = "cuda:0"):
        """
        封装 Transformer，提供训练、保存/加载、评估和预测功能。

        参数:
        ----------
        input_dim : int
            原始输入维度，本例中是 3 (x,y,z)
        embed_dim : int
            Transformer 内部特征维度
        n_heads : int
            多头注意力头数
        num_layers : int
            TransformerEncoder 层数
        dropout : float
            Dropout 比例
        L_in : int
            历史序列长度
        L_out : int
            需要预测的未来步数
        device : str
            设备字符串，如 "cuda:0" 或 "cpu"
        """
        # 存储参数
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.L_in = L_in
        self.L_out = L_out
        self.device = torch.device(device)

        # 实例化 Transformer 模型并移动到指定设备
        self.model = Transformer(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            L_in=L_in,
            L_out=L_out,
        ).to(self.device)
        # 如果想加载预训练权重，可以在这里调用 self.load_model(path)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        对外暴露的预测接口。

        输入:
            x: torch.Tensor, 形状 (B, L_in, 3)

        输出:
            pred: torch.Tensor, 形状 (B, L_out, 3)
        """
        self.model.eval()
        x = x.to(self.device)  # 确保输入在同一个设备上
        with torch.no_grad():
            raw_out = self.model(x)                 # (B, L_out * 3)
            B = raw_out.shape[0]
            pred = raw_out.view(B, self.L_out, -1)  # (B, L_out, 3)
            return pred.cpu()

    def train(self,
            train_loader,
            val_loader,
            num_epochs: int = 50,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            step_size: int = 10,
            gamma: float = 0.5,
            save_root: str = "./checkpoints",
            args=None):
        """
        训练主循环，包括训练、验证、学习率调度、以及模型/配置保存。

        参数:
        ----------
        train_loader : DataLoader
            训练集的 DataLoader，返回 (batch_x, batch_y)，
            batch_x 形状 (B, L_in, 3)，batch_y 形状 (B, L_out, 3)
        val_loader : DataLoader
            验证集的 DataLoader，返回 (val_x, val_y)，
            val_x 形状 (B, L_in, 3)，val_y 形状 (B, L_out, 3)
        num_epochs : int
            训练轮数
        lr : float
            初始学习率
        weight_decay : float
            Adam 优化器的权重衰减
        step_size : int
            StepLR 每隔多少个 epoch 下降一次学习率
        gamma : float
            StepLR 每次下降时乘以的因子
        save_root : str
            用于保存模型和配置的根目录
        args : Namespace 或 类似对象
            包含训练配置的参数（可以是 argparse.Namespace，也可以是你自定义的对象）
            如果传入了 args，将会保存到 config.json 中。
        """
        # 损失函数、优化器和调度器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # 为本次训练创建带时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_root, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # 如果传入 args，就把它保存为 config.json
        if args is not None:
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(vars(args), f, indent=4)

        # 训练主循环
        for epoch in range(1, num_epochs + 1):
            # ========== 训练 ==========
            self.model.train()
            total_train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)  # (B, L_in, 3)
                batch_y = batch_y.to(self.device)  # (B, L_out, 3)

                optimizer.zero_grad()
                # 模型预测输出 (B, L_out, 3)
                raw_pred = self.model(batch_x)  # (B, L_out * 3)
                B = raw_pred.shape[0]
                pred = raw_pred.view(B, self.L_out, -1)  # (B, L_out, 3)

                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * batch_x.size(0)

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # ========== 验证 ==========
            self.model.eval()
            total_val_loss = 0.0
            total_diff_loss = 0.0

            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.to(self.device)
                    val_y = val_y.to(self.device)

                    raw_pred = self.model(val_x)          # (B, L_out * 3)
                    Bv = raw_pred.shape[0]
                    pred_v = raw_pred.view(Bv, self.L_out, -1)  # (B, L_out, 3)

                    # MSE Loss
                    loss_val = criterion(pred_v, val_y)
                    total_val_loss += loss_val.item() * Bv

                    # 计算平均速度预测与最后一步速度预测之差
                    for i in range(Bv):
                        true_velocity = val_y[i, -1, :]                   # (3,)
                        avg_velocity = torch.mean(val_x[i, :5, :], dim=0)  # (3,)
                        diff_loss = torch.norm(true_velocity - avg_velocity).item()
                        total_diff_loss += diff_loss

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            avg_diff_loss = total_diff_loss / len(val_loader.dataset)

            # 每 10 个 epoch 或第 1 个 epoch 打印一次日志
            if epoch == 1 or epoch % 10 == 0:
                lr_current = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:03d} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f} | "
                    f"Diff Loss: {avg_diff_loss:.6f} | "
                    f"LR: {lr_current:.2e}")

        # ========== 训练结束后保存模型 ==========
        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"\n训练完成。模型权重已保存到 → {model_path}")

        return save_dir  # 返回保存目录，以便后续加载

    def load_model(self, model_path: str):
        """
        从指定的路径加载模型权重。

        参数:
        ----------
        model_path : str
            保存的 state_dict 文件路径 (通常是 .pth)
        map_location : str 或 torch.device, 可选
            如果需要将模型加载到特定 device，就传入 "cpu" 或 "cuda:0" 等。
            如果为 None，则使用默认的 self.device。
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"已从 {model_path} 加载模型权重，设备：{self.device}")

    def evaluate(self, data_loader) -> dict:
        """
        在给定的数据集上评估模型，包括 MSE Loss 和平均速度差异损失。

        参数:
        ----------
        data_loader : DataLoader
            用于评估的数据集 DataLoader，返回 (x, y)，
            x 形状 (B, L_in, 3)，y 形状 (B, L_out, 3)

        返回:
        ----------
        metrics : dict
            包含 'mse_loss' 和 'avg_diff_loss' 两个键，对应验证集上的平均值
        """
        criterion = nn.MSELoss()
        self.model.eval()
        total_loss = 0.0
        total_diff = 0.0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                raw_pred = self.model(x)                 # (B, L_out * 3)
                B = raw_pred.shape[0]
                pred = raw_pred.view(B, self.L_out, -1)  # (B, L_out, 3)

                loss = criterion(pred, y)
                total_loss += loss.item() * B

                for i in range(B):
                    true_velocity = y[i, -1, :]                   # (3,)
                    avg_velocity = torch.mean(x[i, :5, :], dim=0)  # (3,)
                    diff_loss = torch.norm(true_velocity - avg_velocity).item()
                    total_diff += diff_loss

        mse_loss = total_loss / len(data_loader.dataset)
        avg_diff_loss = total_diff / len(data_loader.dataset)

        return {
            "mse_loss": mse_loss,
            "avg_diff_loss": avg_diff_loss
        }