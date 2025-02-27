import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa

from sae import SAE, TopKSAE

def set_all_seeds(seed): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_duts3r(duts3r_ckpt_path): 
    duts3r = AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')
    ckpt = torch.load(duts3r_ckpt_path, map_location=device)
    duts3r.load_state_dict(ckpt['model'], strict=False))
    del ckpt
    return duts3r
    
@hydra.main(version_base=None, config_path="configs", config_name="main")    
def train_sae(cfg): 
    
    
    dataloader = build_dataset(cfg.train_dataset, cfg.batch_size, cfg.num_workers, test=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    duts3r = load_duts3r(cfg.duts3r_ckpt_path)
    duts3r = duts3r.to(device)    

    # ToDo : Choose hidden_dim based on selected layer
    sae = SAE(hidden_dim, int(cfg.sparsity_coeff*hidden_dim) )
    optimizer = torch.nn.optim.Adam(sae.parameters())
    activation = {}
    
    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # ToDo : Choose a layer to apply SAE for 
    hook = duts3r.avgpool.register_forward_hook(getActivation('activation'))
    writer = SummaryWriter(cfg.log_path)
    recon_loss = nn.MSELoss()

    for epoch_idx in range(n_epochs): 
        for batch_idx, batch in enumerate(dataloader): 
            
            glob_iter = batch_idx + epoch_idx*len(dataloader)
            with torch.inference_mode(): 
              _ = duts3r(X)
            act = activation['activation']
            
            optimizer.zero_grad()
            act_recon, act_encode = sae(act)
            loss = recon_loss(act, act_recon) + cfg.reg_coeff * act_encode.sum()
            loss.backward() 
            optimizer.step()

            writer.add_scalar("Train/loss_iter", loss.item(),  glob_iter)
            writer.add_scalar("Train/epoch_iter", epoch_idx + 1, glob_iter)

            if (glob_iter % cfg.ckpt_iter) == 0: 
                torch.save({
                'epoch': epoch_idx,
                'iter' : batch_idx,
                'glob_iter' : glob_iter,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, cfg.ckpt_path)

                
if name == '__main__': 
    train_sae()