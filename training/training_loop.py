import torch
import torch.nn.functional as F

EPOCHS = 100

def train(model, loader, optim, device='cpu'):
    model.train()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'


    for epoch in range(EPOCHS):

        for window in loader:
            window = window.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z_hat, z = model(window)  # run the forward pass

                loss = F.smooth_l1_loss(z_hat, z)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            model.update_target_params()