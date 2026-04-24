# import 
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# FNN
class FNN(nn.Module):
    def __init__(self, layers, device):
        super(FNN, self).__init__()
        self.func = nn.Tanh  # activation function
        self.layers = layers
        self.device = device
        self.model = self.create_model().to(self.device)

    def create_model(self):
        layers_list = []
        for i in range(len(self.layers) - 1):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                layers_list.append(self.func())
        return nn.Sequential(*layers_list)

    def forward(self, X):
        out = self.model(X)
        return out
    
# Normal PINN
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

class normal_pinn():
    def __init__(self, layers, device):
        self.device = device
        self.layers = layers
        self.net = FNN(layers, device)
            
        self.loss_history = []
        self.loss_compute = None
        
        # --- 新增：定义对抗训练超参数 ---
        # 使用 log 空间初始化 alpha, beta 保证其经过 softplus 后为正
        # gamma 初始化为 0 (对应中位数)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
        self.raw_beta  = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float32))
        self.raw_gamma = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
        
        # --- 新增：主网络优化器 ---
        self.optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.hyper_wd = hyperwd
        # --- 新增：超参数单独优化器 (单独设置 lr，通常大一点，例如 1e-2) ---
        self.optimizer_hyper = torch.optim.Adam(
            [
            {'params': [self.raw_alpha ],'lr':1e-3, 'weight_decay':self.hyper_wd["alpha_wd"]},
            {'params': [self.raw_beta], 'lr': 1e-3, 'weight_decay': self.hyper_wd["beta_wd"]},
            {'params': [self.raw_gamma], 'lr': 1e-5},
            ]
        )

        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.iter = 0
        self.epochs = 0
        self.loss_history = []

    def get_adv_hyperparams(self):
        
        alpha_min = 0
        alpha_max = 10.0
        # alpha = torch.nn.functional.softplus(self.raw_alpha)
        # alpha = alpha_min + (alpha_max-alpha_min) * torch.tanh(alpha)
        alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(self.raw_alpha - 5.0)   # 用偏移的sigmoid函数，保证后期的权重能够下降或者近似下降到最小值
        
        beta_min = 0.1
        beta_max = 3.0
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(self.raw_beta - 5.0)
        
        gamma_min = -3.0
        gamma_max = 3.0
        gamma = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(self.raw_gamma)
        # 3. 应用梯度翻转
        return GradientReversal.apply(alpha), GradientReversal.apply(beta), GradientReversal.apply(gamma)

    def configure_scheduler(self, scheme="none"):
        scheme = (scheme or "none").lower()
        self.scheduler_adam = None
        self.scheduler_lbfgs = None
        if scheme == "exp":
            self.scheduler_adam = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_adam, gamma=0.9
            )
            self.scheduler_lbfgs = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer_lbfgs, gamma=0.9
            )
        elif scheme == "plateau":
            self.scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_adam, mode="min", factor=0.9, patience=5000, min_lr=1e-6
            )
            self.scheduler_lbfgs = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_lbfgs, mode="min", factor=0.9, patience=400, min_lr=1e-6
            )

    def net_u(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        u = self.net(inputs)
        return u

    def loss_compute_updata(self, loss_compute, X_ic_train, X_bc_train, X_f_train, device, config=None):
        self.loss_compute = loss_compute(
            self,
            X_ic_train,
            X_bc_train,
            X_f_train,
            device,
            config,
        )

    def loss(self):
        loss_pde, loss_ic, loss_bc = self.loss_compute.loss_compute()
        total_loss = self.loss_compute.weighted_loss(loss_pde, loss_ic, loss_bc)
        return total_loss, loss_pde, loss_ic, loss_bc

    def train(self,
              epochs=1000,
              print_every=100,
              opt_type='adam',
              scheduler='none',
              plot_judge='No',
              plot_type='none',
              plot_every=None,
              save_dir=None
             ):
        self.epochs = epochs
        if plot_every is None:
            plot_every = epochs
        opt_type = (opt_type or "adam").lower()
        if opt_type not in {"adam", "lbfgs"}:
            raise ValueError(f"Unsupported opt_type '{opt_type}'. Expected 'adam' or 'lbfgs'.")
        self.configure_scheduler(scheduler)
        self.net.train()

        if opt_type == "adam":
            optimizer = self.optimizer_adam
            scheduler_obj = self.scheduler_adam
            
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                self.optimizer_hyper.zero_grad() # 新增：清空超参数梯度
                
                total_loss, loss_pde, loss_ic, loss_bc = self.loss()
                total_loss.backward()
                
                optimizer.step()
                if epoch > 30000:
                    self.optimizer_hyper.step()    # 延迟启动
                    with torch.no_grad():
                        self.raw_alpha.clamp_(min=0.0, max=10.0) 
                        self.raw_beta.clamp_(min=0.0, max=10.0)
                        self.raw_gamma.clamp_(min=-5.0, max=5.0)  
                
                if scheduler_obj is not None:
                    if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler_obj.step(total_loss.item())
                    else:
                        scheduler_obj.step()
                self.iter += 1
                eff_alpha, eff_beta, eff_gamma = self.get_adv_hyperparams()
                history_entry = {
                    "epoch": self.iter,
                    "total": total_loss.item(),
                    "pde": loss_pde.item(),
                    "ic": loss_ic.item(),
                    "bc": loss_bc.item(),
                    "stage": "adam",
                    "alpha": eff_alpha.item(),
                    "beta": eff_beta.item(),
                    "gamma": eff_gamma.item(),
                }
                self.loss_history.append(history_entry)
                
                if print_every and epoch % print_every == 0:
                    
                    current_alpha = eff_alpha.item()
                    current_beta  = eff_beta.item()
                    current_gamma = eff_gamma.item()
                    
                    print(
                        f"[Adam] Epoch {self.iter}: total={total_loss.item():.4e}, "
                        f"pde={loss_pde.item():.4e}, ic={loss_ic.item():.4e}, bc={loss_bc.item():.4e} | "
                        f"Adv[α={current_alpha:.2f}, β={current_beta:.2f}, γ={current_gamma:.4f}]"
                    )
                if plot_judge != 'No' and epoch % plot_every == 0:
                        self.loss_compute.point_scater(plot_type=plot_type, save_dir=save_dir, name=f'[Adam]epoch[{epoch}:{self.epochs}]')
        else:
            optimizer = self.optimizer_lbfgs
            scheduler_obj = self.scheduler_lbfgs
            
            def closure():
                optimizer.zero_grad()
                self.optimizer_hyper.zero_grad() 
                
                total_loss, _, _, _ = self.loss()
                total_loss.backward()
                return total_loss

            for epoch in range(1, self.epochs + 1):
                optimizer.step(closure)
                
                self.optimizer_hyper.zero_grad()
                total_loss, loss_pde, loss_ic, loss_bc = self.loss()
                total_loss.backward()
                self.optimizer_hyper.step()

                if scheduler_obj is not None:
                    if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler_obj.step(total_loss.item())
                    else:
                        scheduler_obj.step()
                self.iter += 1
                history_entry = {
                    "epoch": self.iter,
                    "total": total_loss.item(),
                    "pde": loss_pde.item(),
                    "ic": loss_ic.item(),
                    "bc": loss_bc.item(),
                    "stage": "lbfgs",
                    "alpha": eff_alpha.item(),
                    "beta": eff_beta.item(),
                    "gamma": eff_gamma.item(),
                }
                self.loss_history.append(history_entry)

                if print_every and epoch % print_every == 0:
                    current_alpha = torch.nn.functional.softplus(self.raw_alpha).item()
                    current_beta  = torch.nn.functional.softplus(self.raw_beta).item()
                    current_gamma = self.raw_gamma.item()
                    
                    print(
                        f"[LBFGS] Epoch {self.iter}: total={total_loss.item():.4e}, "
                        f"pde={loss_pde.item():.4e}, ic={loss_ic.item():.4e}, bc={loss_bc.item():.4e} | "
                        f"Adv[α={current_alpha:.2f}, β={current_beta:.2f}, γ={current_gamma:.4f}]"
                    )
                if plot_judge != 'No' and epoch % plot_every == 0:
                        self.loss_compute.point_scater(plot_type=plot_type, save_dir=save_dir, name=f'[LBFGS]epoch[{epoch}:{self.epochs}]')

        return self.loss_history

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)
        x = X_tensor[:, 0:1]
        t = X_tensor[:, 1:2]
        self.net.eval()
        with torch.no_grad():
            u = self.net_u(x, t)
        return u.cpu().numpy()
        
    def save_model(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        state = {
            'net_state_dict': self.net.state_dict(),
            'optimizer_adam_state_dict': self.optimizer_adam.state_dict(),
            'optimizer_lbfgs_state_dict': self.optimizer_lbfgs.state_dict(),
            # 新增：保存超参数和其优化器状态
            'raw_alpha': self.raw_alpha,
            'raw_beta': self.raw_beta,
            'raw_gamma': self.raw_gamma,
            'optimizer_hyper_state_dict': self.optimizer_hyper.state_dict(),
            'loss_history': self.loss_history,
            'iter': self.iter,
            'layers': self.layers,
        }
        torch.save(state, file_path)
        print(f"Model saved successfully to {file_path}")

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        checkpoint = torch.load(file_path, map_location=self.device)
        
        if 'layers' in checkpoint and checkpoint['layers'] != self.layers:
             print("Warning: Loaded model layers do not match current model layers.")

        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer_adam.load_state_dict(checkpoint['optimizer_adam_state_dict'])
        self.optimizer_lbfgs.load_state_dict(checkpoint['optimizer_lbfgs_state_dict'])
        
        # 新增：加载超参数
        if 'raw_alpha' in checkpoint:
            self.raw_alpha.data = checkpoint['raw_alpha'].data
            self.raw_beta.data = checkpoint['raw_beta'].data
            self.raw_gamma.data = checkpoint['raw_gamma'].data
            self.optimizer_hyper.load_state_dict(checkpoint['optimizer_hyper_state_dict'])
        
        self.loss_history = checkpoint['loss_history']
        self.iter = checkpoint['iter']
        print(f"Model loaded successfully from {file_path}, current iter: {self.iter}")


# LossCompute
class LossCompute():
    def __init__(
        self,
        model,
        X_ic_train,
        X_bc_train,
        X_f_train,
        device,
        config=None,
    ):
        self.model = model
        self.device = device
        # loss weight design
        self.config = config or {}
        default_weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
        custom_weights = self.config.get("weights", {})
        self.weights = {**default_weights, **custom_weights}
        default_pointwise_weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
        custom_pointwise_weights = self.config.get('pointwise_weights',{})
        self.pointwise_weights = {**default_pointwise_weights,**custom_pointwise_weights}
    
        self.x_ic = torch.tensor(X_ic_train[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_ic = torch.tensor(X_ic_train[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        self.u_ic = torch.tensor(X_ic_train[:, 2:3], dtype=torch.float32, device=device)

        self.x_bc = torch.tensor(X_bc_train[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_bc = torch.tensor(X_bc_train[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)

        self.x_f = torch.tensor(X_f_train[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
        self.t_f = torch.tensor(X_f_train[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
        eps = torch.random
        self.residual = {
            'pde': None,
            'ic': None,
            'bc': None,
        }

    def gradient(self, func, var, order=1):
        if order == 1:
            return torch.autograd.grad(
                func,
                var,
                grad_outputs=torch.ones_like(func),
                retain_graph=True,
                create_graph=True,
            )[0]
        else:
            out = self.gradient(func, var)
            return self.gradient(out, var, order - 1)
        
    def _resolve_weight(self, key):
        weight = self.weights.get(key, 1.0)
        if callable(weight):
            weight = weight()
        if not torch.is_tensor(weight):
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        return weight
    
    def _resolve_pointwise_weight(self, key, t_ref):
        weight = self.pointwise_weights.get(key, 1.0)
        if callable(weight):
            weight = weight()
        if not torch.is_tensor(weight):
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        if weight.ndim == 0 and t_ref is not None:
            weight = weight.expand_as(t_ref)
        return weight
    
    def loss_pde(self):
        self.x_f.requires_grad_(True)
        self.t_f.requires_grad_(True)
        x = self.x_f
        t = self.t_f
        u = self.model.net_u(x, t)
        u_t = self.gradient(u, t)
        u_x = self.gradient(u, x)
        u_xx = self.gradient(u_x, x)
        residual = u_t - 1e-4 * u_xx + 5.0 * u**3 - 5.0 * u
        self.residual['pde'] = residual.abs().detach()
        weight = self._resolve_pointwise_weight('pde', t)
        # print(f'pde pointwise weight:{weight.mean()}')
        residual = weight * residual
        return torch.mean(residual**2)

    def loss_bc(self):
        self.x_bc.requires_grad_(True)
        x = self.x_bc
        t = self.t_bc
        n_half = x.shape[0] // 2
        x_left, x_right = x[:n_half], x[n_half:]
        t_left, t_right = t[:n_half], t[n_half:]

        u_left = self.model.net_u(x_left, t_left)
        u_right = self.model.net_u(x_right, t_right)
        u_x_left = self.gradient(u_left, x_left)
        u_x_right = self.gradient(u_right, x_right)
        residual_u = (u_left - u_right).abs().detach()
        residual_du = (u_x_left - u_x_right).abs().detach()
        self.residual['bc'] = torch.vstack([residual_u, residual_du])
        weight = self._resolve_pointwise_weight('bc', t)
        # print(f'bc pointwise weight:{weight.mean()}')
        weight_u = weight[:n_half]
        weight_du = weight[n_half:]
        u_bc_err = (weight_u * (u_left - u_right)) ** 2
        du_bc_err = (weight_du * (u_x_left - u_x_right)) ** 2
        loss_u = torch.mean(u_bc_err)
        loss_du = torch.mean(du_bc_err)
        return loss_u + loss_du

    def loss_ic(self):
        u_pred = self.model.net_u(self.x_ic, self.t_ic)
        residual = (u_pred - self.u_ic).abs().detach()
        self.residual['ic'] = residual
        weight = self._resolve_pointwise_weight('ic',self.t_ic)
        u_ic_err = weight*(u_pred-self.u_ic)**2
        loss_ic = torch.mean(u_ic_err)
        return loss_ic

    def weighted_loss(self, loss_pde, loss_ic, loss_bc):
        w_pde = self._resolve_weight("pde")
        w_ic = self._resolve_weight("ic")
        w_bc = self._resolve_weight("bc")
        return w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

    def loss_compute(self):
        loss_pde = self.loss_pde()
        loss_bc = self.loss_bc()
        loss_ic = self.loss_ic()
        return loss_pde, loss_ic, loss_bc
    
    def up_data(self,X_f_new):  
        # use in resample to updata the sample
        self.x_f = torch.tensor(X_f_new[:, 0:1], dtype=torch.float32, device=self.device)
        self.t_f = torch.tensor(X_f_new[:, 1:2], dtype=torch.float32, device=self.device)

        self.x_f.requires_grad_(True)
        self.t_f.requires_grad_(True)

    def updata_config(self,config):
        self.config = config
        default_weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
        custom_weights = self.config.get("weights", {})
        self.weights = {**default_weights, **custom_weights}
        default_pointwise_weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
        custom_pointwise_weights = self.config.get('pointwise_weights',{})
        self.pointwise_weights = {**default_pointwise_weights,**custom_pointwise_weights}

    def point_scater(self,
                     plot_type='none',
                     cmap='viridis',f_s=1, ic_s=1, bc_s=1,
                     save_dir=None,
                     name=None,
                    ):
        mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        })
        plt.figure(figsize=(8,6))
        t_f = self.t_f.detach().cpu().numpy().ravel()
        x_f = self.x_f.detach().cpu().numpy().ravel()
        t_ic = self.t_ic.detach().cpu().numpy().ravel()
        x_ic = self.x_ic.detach().cpu().numpy().ravel()
        t_bc = self.t_bc.detach().cpu().numpy().ravel()
        x_bc = self.x_bc.detach().cpu().numpy().ravel()

        if plot_type == 'weighted':
            w_f = self._resolve_pointwise_weight('pde', self.t_f)
            w_ic = self._resolve_pointwise_weight('ic', self.t_ic)
            w_bc = self._resolve_pointwise_weight('bc', self.t_bc)

            w_f = w_f.detach().cpu().numpy().ravel() if torch.is_tensor(w_f) else np.asarray(w_f).ravel()
            w_ic = w_ic.detach().cpu().numpy().ravel() if torch.is_tensor(w_ic) else np.asarray(w_ic).ravel()
            w_bc = w_bc.detach().cpu().numpy().ravel() if torch.is_tensor(w_bc) else np.asarray(w_bc).ravel()

            arrs = [a for a in (w_f, w_ic, w_bc) if a is not None and a.size > 0]
            if not arrs:
                raise RuntimeError("没有可用的权重用于绘图。")
            vmin, vmax = min(a.min() for a in arrs), max(a.max() for a in arrs)

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            plt.scatter(t_f, x_f, c=w_f, cmap=cmap, norm=norm, s=f_s, marker='o', alpha=0.9, label='collocation')
            plt.scatter(t_ic, x_ic, c=w_ic, cmap=cmap, norm=norm, s=ic_s, marker='x', linewidths=1.0, label='IC')
            plt.scatter(t_bc, x_bc, c=w_bc, cmap=cmap, norm=norm, s=bc_s, marker='x', linewidths=1.0, label='BC')

            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(np.concatenate([a.ravel() for a in arrs]))
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('w')
        else:
            plt.scatter(t_f, x_f, s=f_s, marker='o', alpha=0.9, label='collocation')
            plt.scatter(t_ic, x_ic, s=ic_s, marker='x', color='k', linewidths=1.0, label='IC')
            plt.scatter(t_bc, x_bc, s=bc_s, marker='x', color='r', linewidths=1.0, label='BC')
        plt.xlabel('t'); plt.ylabel('x')
        plt.title('Points Scatter (w mapped to color)')
        plt.legend(loc='best')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{name}sample_location&weight.png"), dpi=300)
        plt.show()
        plt.close()

# Loss History Function
def plot_loss_history(training_history, title="Training loss history", save_path=None):
    if not training_history:
        raise ValueError("training_history 为空，无法绘图。")
    epochs = np.array([entry["epoch"] for entry in training_history])
    loss_total = [entry["total"] for entry in training_history]
    loss_pde = [entry["pde"] for entry in training_history]
    loss_ic = [entry["ic"] for entry in training_history]
    loss_bc = [entry["bc"] for entry in training_history]
    lbfgs_start = next((entry["epoch"] for entry in training_history if entry.get("stage") == "lbfgs"), None)

    plt.figure(figsize=(8, 4))
    plt.semilogy(epochs, loss_total, label="total")
    plt.semilogy(epochs, loss_pde, label="pde")
    plt.semilogy(epochs, loss_ic, label="ic")
    plt.semilogy(epochs, loss_bc, label="bc")

    if lbfgs_start is not None:
        plt.axvline(lbfgs_start, color="k", linestyle="--", alpha=0.4)
        plt.text(lbfgs_start, max(loss_total), " LBFGS", va="bottom", ha="left", fontsize=9, color="k")

    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

# Visualize Function
def allen_cahn_reference_solution(x_values, t_values, initial_condition_fn, diffusion=1e-4, reaction=5.0):
    x_values = np.asarray(x_values)
    t_values = np.asarray(t_values)
    dx = x_values[1] - x_values[0]
    dt = t_values[1] - t_values[0]
    Nx = x_values.size
    Nt = t_values.size
    u = np.zeros((Nx, Nt), dtype=np.float64)
    u[:, 0] = initial_condition_fn(x_values[:, None]).ravel()
    for n in range(1, Nt):
        u_prev = u[:, n - 1]
        u_xx = (np.roll(u_prev, -1) - 2 * u_prev + np.roll(u_prev, 1)) / dx ** 2
        reaction_term = -reaction * u_prev ** 3 + reaction * u_prev
        u[:, n] = u_prev + dt * (diffusion * u_xx + reaction_term)
    return u

def plot_solution_comparison(model, x_plot, t_plot, initial_condition_fn, title_prefix="PINN", save_dir=None, t_slices=None):
    T_grid, X_grid = np.meshgrid(t_plot, x_plot)
    XT = np.stack([X_grid.ravel(), T_grid.ravel()], axis=1)
    U_pred = model.predict(XT).reshape(X_grid.shape)
    U_ref = allen_cahn_reference_solution(x_plot, t_plot, initial_condition_fn)
    U_error = U_pred - U_ref
    abs_error = np.abs(U_error)
    l2_error = np.sqrt(np.mean(U_error ** 2))
    max_error = np.max(abs_error)
    print(f"{title_prefix} -> L2 error: {l2_error:.4e}, max abs error: {max_error:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    pcm0 = axes[0].pcolormesh(t_plot, x_plot, U_pred, shading="auto", cmap="RdBu_r")
    axes[0].set_title(f"{title_prefix} prediction")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(pcm0, ax=axes[0], shrink=0.85, label="u(x, t)")

    pcm1 = axes[1].pcolormesh(t_plot, x_plot, U_ref, shading="auto", cmap="RdBu_r")
    axes[1].set_title("Reference solution")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(pcm1, ax=axes[1], shrink=0.85, label="u(x, t)")

    pcm2 = axes[2].pcolormesh(t_plot, x_plot, abs_error, shading="auto", cmap="viridis")
    axes[2].set_title(f"Absolute error |{title_prefix} - Ref|")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(pcm2, ax=axes[2], shrink=0.85, label="|error|")

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()

    slice_times = t_slices or [0.0, 0.25, 0.5, 0.75, 1.0]
    slice_fig, slice_axes = plt.subplots(1, len(slice_times), figsize=(4 * len(slice_times), 4), sharey=True)
    if len(slice_times) == 1:
        slice_axes = [slice_axes]
    for ax, ti in zip(slice_axes, slice_times):
        idx_t = int(np.argmin(np.abs(t_plot - ti)))
        ax.plot(x_plot, U_pred[:, idx_t], label=f"{title_prefix} t={t_plot[idx_t]:.2f}")
        ax.plot(x_plot, U_ref[:, idx_t], linestyle="--", label=f"Ref t={t_plot[idx_t]:.2f}")
        ax.set_xlabel("x")
        ax.set_title(f"t={t_plot[idx_t]:.2f}")
        ax.grid(alpha=0.3)
    slice_axes[0].set_ylabel("u(x, t)")
    slice_axes[-1].legend(loc="upper right", fontsize=8)
    slice_fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = title_prefix.lower().replace(" ", "_")
        fig.savefig(os.path.join(save_dir, f"{base}_surface.png"), dpi=300)
        slice_fig.savefig(os.path.join(save_dir, f"{base}_slices.png"), dpi=300)

    plt.show()
    plt.close(fig)
    plt.close(slice_fig)

# Weight Function
def omega_weight(model, target):
    device = getattr(model, 'device', torch.device('cpu'))
    target = (target or '').lower()
    parts = target.split(':', 1)
    base_target = parts[0]

    def factory():
        loss_compute = getattr(model, "loss_compute", None)
        if loss_compute is None:
            raise AttributeError("LossCompute 尚未初始化，无法计算权重。")

        residual_store = loss_compute.residual.get(base_target)
        if residual_store is None:
                    raise RuntimeError(f"尚未为 {base_target} 计算 residual, 无法生成权重。请先执行对应的 loss 计算。")
        alpha, beta, gamma = model.get_adv_hyperparams()
        
        with torch.no_grad():
            residual_abs = residual_store.abs().detach()
            r_median = torch.median(residual_abs)
            r_mad = torch.median(torch.abs(residual_abs - r_median)) + 1e-8
            r_normalized = (residual_abs - r_median) / r_mad

        attention_score = torch.sigmoid((r_normalized - gamma) * beta)
        s = 1.0 + alpha * attention_score
        w = s / (s.mean() + 1e-8)
        return w

    return factory

# Save Log Function
def save_log(save_dir,training_history):
    md_path = f"{save_dir}/train_epochs.md"
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 训练 Epoch 日志\n\n")
        f.write("| epoch | stage | total | pde | bc |\n")
        f.write("|---:|:---:|---:|---:|---:|\n")
        for e in training_history:
            if e['epoch']%10000 == 0:
                f.write(f"| {e['epoch']} | {e.get('stage','')} | {e['total']:.4e} | {e['pde']:.4e} | {e['bc']:.4e} |\n")
    print(f"Saved epoch log to {md_path}")
            
# Sampling
np.random.seed(1234)
torch.manual_seed(1234)
rng = np.random.default_rng(1234)

def initial_condition(x):
    return x ** 2 * np.cos(np.pi * x)

def latin_hypercube_sampling(bounds, n_samples, rng):
    """Simple LHS sampler for given bounds [(low, high), ...]."""
    dim = len(bounds)
    result = np.zeros((n_samples, dim))
    cut = np.linspace(0, 1, n_samples + 1)
    for i, (low, high) in enumerate(bounds):
        u = rng.random(n_samples)
        pts = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(pts)
        result[:, i] = low + pts * (high - low)
    return result

N_ic = 512
x_ic = latin_hypercube_sampling([(-1.0, 1.0)], N_ic, rng)
t_ic = np.zeros((N_ic, 1))
u_ic = initial_condition(x_ic)
X_ic_train = np.hstack([x_ic, t_ic, u_ic])

N_bc = 100
t_bc = latin_hypercube_sampling([(0.0, 1.0)], N_bc, rng)
x_left = -np.ones_like(t_bc)
x_right = np.ones_like(t_bc)
X_bc_left = np.hstack([x_left, t_bc])
X_bc_right = np.hstack([x_right, t_bc])
X_bc_train = np.vstack([X_bc_left, X_bc_right])

N_f = 25600
X_f = latin_hypercube_sampling([(-1.0, 1.0), (0.0, 1.0)], N_f, rng)
x_f = X_f[:, 0:1]
t_f = X_f[:, 1:2]
X_f_train = np.hstack([x_f, t_f])

#==================================#

# Weight Train
hyper_wd_ls = [1e-5,1e-6,1e-7]
epochs = 300000
layers = [2,128,128,128,128,1]
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

for wd_a in hyper_wd_ls:
    for wd_b in hyper_wd_ls:
        hyperwd = {"alpha_wd":wd_a,"beta_wd":wd_b}
        model = f"{wd_a}#{wd_b}"
        print(f'current train model:{model}')
        save_dir = 'AC/test/'+model

        pinn_curr = normal_pinn(
            layers = layers,
            device = device,
            hyperwd=hyperwd,
        )
        pinn_curr.loss_compute_updata(LossCompute,X_ic_train,X_bc_train,X_f_train,device)

        pde_weight = omega_weight(pinn_curr, "pde")
        loss_config = {'weights':{'pde':1,'ic':100},
                    'pointwise_weights':{'pde':pde_weight}
                    }
        pinn_curr.loss_compute.updata_config(loss_config)
        print("===TRAINING MODEL===")
        training_history = pinn_curr.train(
            epochs=epochs,
            print_every=10000,
            opt_type= 'adam',
            scheduler='plateau',
            plot_judge = 'Yes',
            plot_every = 10000,
            plot_type = 'weighted',
            save_dir=f"{save_dir}/sample_weight&location",
        )
        print("===SAVE MODEL===")
        save_path = f"{save_dir}/{model}.pth"
        pinn_curr.save_model(save_path)

        plot_loss_history(
            training_history,
            title="AC Sigmoid Adaptive Loss History",
            save_path=f"{save_dir}/loss.png",
        )
        x_plot = np.linspace(-1, 1, 200)
        t_plot = np.linspace(0, 1, 200)
        plot_solution_comparison(
            model=pinn_curr,
            x_plot=x_plot,
            t_plot=t_plot,
            initial_condition_fn=initial_condition,
            title_prefix="AC Sigmoid Adaptive",
            save_dir= save_dir,
        )
        print("===SAVE LOGS===")
        save_log(save_dir,training_history)
        print("============================")