import torch
import torch.nn as nn
import hydra
from torch.utils.tensorboard import SummaryWriter
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
import datetime

from tqdm import tqdm
from sae import VanillaSAE

def set_all_seeds(seed): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dust3r(dust3r_ckpt_path, device): 
    return AsymmetricCroCo3DStereo.from_pretrained(dust3r_ckpt_path, device=device)
    
@hydra.main(version_base=None, config_path="configs", config_name="main")    
def train_sae(cfg): 

    dataloader = get_data_loader(
	    cfg.train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_mem=True,
        shuffle=True,
        drop_last=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dust3r = load_dust3r(cfg.dust3r_ckpt_path, device)
    dust3r = dust3r.to(device)

    hidden_dim = dust3r.enc_norm.normalized_shape[0]
    print(hidden_dim)

    cfg_sae = {
        'l1_coeff' : 0.001,
        'act_size' : hidden_dim, 
        'dict_size' : 4 * hidden_dim, 
        'seed' : 42, 
        'device' : device, 
        'dtype' : torch.float32, 
    }

    sae = VanillaSAE(cfg_sae)
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # ToDo : Choose a layer to apply SAE for
    writer = SummaryWriter(cfg.log_path)
    recon_loss = nn.MSELoss()

    for epoch_idx in range(cfg.num_epochs): 
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)): 
            
            glob_iter = batch_idx + epoch_idx*len(dataloader)
            ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)
            with torch.inference_mode(): 
                view1, view2 = batch
                (shape1, shape2), (feat1, feat2), (pos1, pos2) = dust3r._encode_symmetrized(view1, view2)
            act = torch.concat((feat1, feat2), dim=1)                                                                                                                                                                                                            
            optimizer.zero_grad()
            out = sae(act)
            # loss = recon_loss(act, act_recon) + cfg.reg_coeff * act_encode.abs().sum()
            out["loss"].backward() 
            # print(out["loss"].item())
            # torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            """
            {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            }
            """

            writer.add_scalar("Train/loss_iter", out["loss"].item(),  glob_iter)
            writer.add_scalar("Train/epoch_iter", epoch_idx + 1, glob_iter)
            writer.add_scalar("Train/l2", out["l2_loss"].item(), glob_iter)
            writer.add_scalar("Train/l1", out["l1_loss"], glob_iter)
            writer.add_scalar("Train/nnz", (out["feature_acts"].abs() > 1e-6 ).sum() / int(cfg_sae['dict_size']), glob_iter)
            energy = torch.sum(out["feature_acts"].detach().cpu()**2)
            writer.add_scalar("Train/energy", energy.item(),  glob_iter)
            for n, p in sae.named_parameters():
                writer.add_scalar("Train/grad_norm_" + n, p.grad.data.norm(2), glob_iter)
            # timestamp = datetime.datetime.now().strftime("%H:%M:%S - %d.%m.%Y")
            if (glob_iter % cfg.ckpt_iter) == 0: 
                torch.save({
                'epoch': epoch_idx,
                'iter' : batch_idx,
                'glob_iter' : glob_iter,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, cfg.ckpt_path)


if __name__ == '__main__': 
    train_sae()
