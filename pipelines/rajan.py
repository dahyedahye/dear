"""Train the base Rajan (AlignedForensics) detector (ResNet-50, aligned pairs).

This produces the pretrained checkpoint that DEAR-r is built on. Most users can
skip this and download the released base checkpoint instead.
"""

import os
import uuid
import tempfile

import tqdm
import wandb
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import torch

from dear.utils import set_seed, EarlyStopping
from dear.detector.rajan_detector import RajanDetector
from dear.dataset.rajan_dataset import load_rajan_dataset
from pipelines.utils import run_inference, compute_test_ap


@hydra.main(config_path="../configs", config_name="rajan", version_base=None)
def main(args):
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(args, resolve=True),
            project='dear', group=args.wandb_group, name=args.wandb_name,
            id=str(uuid.uuid4()), dir=tempfile.mkdtemp(),
        )

    os.makedirs(args.save_path, exist_ok=True)

    detector = RajanDetector(
        lr=args.lr, beta1=args.beta1, weight_decay=args.weight_decay,
        device=args.device, batched_syncing=args.batched_syncing,
    )

    if args.mode == "train":
        train_dataset, val_dataset = load_rajan_dataset(
            args.data_root, args.train_real_types, args.train_fake_types,
            args.val_real_types, args.val_fake_types, transform_cfg=args.transform,
            use_inversions=args.use_inversions, batched_syncing=args.batched_syncing,
            seed=args.seed,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        n_gradient_step = 0
        early_stopping = None

        for epoch in range(0, args.num_epochs):
            # ----- Validation -----
            detector.eval()
            y_true, y_pred = [], []
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x = batch['image'].to(args.device)
                y = batch['target'].to(args.device).float()
                logits = detector.predict(x).squeeze(1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(logits.cpu().numpy().tolist())

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc = balanced_accuracy_score(y_true, y_pred > 0.0)
            auc = roc_auc_score(y_true, y_pred)
            test_ap = compute_test_ap(detector, args)
            print(f"After {epoch} epochs: val acc={acc:.4f}, val auc={auc:.4f}, test AP={test_ap:.2f}")

            if args.use_wandb:
                wandb.log({"val_acc": acc, "val_auc": auc, "test_ap": test_ap,
                           "lr": detector.get_learning_rate()}, step=n_gradient_step)

            # ----- Early stopping -----
            if early_stopping is None:
                early_stopping = EarlyStopping(
                    init_score=acc, patience=args.earlystop_patience,
                    delta=args.earlystop_delta, verbose=True,
                )
            else:
                if early_stopping(acc):
                    print("Validation improved, saving model ...", flush=True)
                    detector.save(f"{args.save_path}/model_best.pth")
                if early_stopping.early_stop:
                    if detector.adjust_learning_rate():
                        print("Learning rate dropped by 10, continue training ...", flush=True)
                        early_stopping.reset_counter()
                    else:
                        print("Early stopping.", flush=True)
                        break

            # ----- Training -----
            detector.train()
            for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
                if args.batched_syncing:
                    real_batch, fake_batch = batch
                    x = (real_batch['image'].to(args.device),
                         fake_batch['image'].to(args.device))
                    y = (real_batch['target'].to(args.device).float(),
                         fake_batch['target'].to(args.device).float())
                else:
                    x = batch['image'].to(args.device)
                    y = batch['target'].to(args.device).float()

                loss = detector.update(x, y)['loss']
                n_gradient_step += 1
                if n_gradient_step % 1000 == 0 and args.use_wandb:
                    wandb.log({"train_loss": loss}, step=n_gradient_step)

            if epoch % args.save_every == 0:
                detector.save(f"{args.save_path}/model_{epoch}.pth")

    elif args.mode == "inference":
        run_inference(detector, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
