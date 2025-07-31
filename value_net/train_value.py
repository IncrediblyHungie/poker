import torch, torchmetrics, tqdm
from torch.utils.data import DataLoader
from config import CFG
from value_net.dataset import CFRDataset
from value_net.model import TransformerValue

def main():
    ds = CFRDataset(CFG["cfr"].storage)
    dl = DataLoader(ds, batch_size=CFG["vnet"].batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = TransformerValue(CFG["vnet"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["vnet"].lr)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["vnet"].fp16)

    for epoch in range(CFG["vnet"].epochs):
        pbar = tqdm.tqdm(dl, desc=f"epoch {epoch}")
        for obs, tgt in pbar:
            obs, tgt = obs.cuda(), tgt.cuda()
            with torch.cuda.amp.autocast(enabled=CFG["vnet"].fp16):
                out = model(obs)
                loss = torch.nn.functional.mse_loss(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            pbar.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), CFG["vnet"].ckpt_dir / f"vnet_{epoch}.pt")

if __name__ == "__main__":
    main()
