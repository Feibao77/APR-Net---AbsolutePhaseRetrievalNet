import os
import time
import datetime
from typing import Union, List

import torch
from torch.utils import data

from src.model import APRNet_full

from train_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler
from dataset import APRDataset

import transforms as T

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = APRDataset(args.data_path, train=True,
                                )
    val_dataset = APRDataset(args.data_path, train=False,
                              )

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn,
                                        drop_last=False)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn,
                                      drop_last=False)

    model = APRNet_full()
    model.to(device)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=2)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_mae = 1.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:

            mae_metric = evaluate(model, val_data_loader, device=device)
            mae_info = mae_metric.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.6f} ")
            # write into txt
            with open(results_file, "a") as f:

                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.6f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.6f}  \n"
                f.write(write_info)

            # save_best
            if current_mae >= mae_info:
                torch.save(save_file,
                           "save_weights/model_best.pth")

        # only save latest 10 epoch weights
        if os.path.exists(
                f"save_weights/model_{epoch - 10}.pth"):
            os.remove(
                f"save_weights/model_{epoch - 10}.pth")

        torch.save(save_file,
                   f"save_weights/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch APR-Net training")

    parser.add_argument("--data-path", default="./Training and Verification datasets", help="APR root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=400, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    main(args)
