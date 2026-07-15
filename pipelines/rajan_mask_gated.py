"""DEAR-r: apply DEAR (Dissect and Prune) to the base Rajan detector.

Loads the pretrained Rajan base detector, dissects its channels with RAD on the
diagnostic inpaint set, bilaterally prunes the extremes, then refines the linear
head on (Rajan aligned pairs + inpaint data).

    python pipelines/rajan_mask_gated.py mode=train \\
        exclude_top_ratio=0.3 exclude_bottom_ratio=0.3
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
from dear.detector.rajan_mask_gated_detector import RajanMaskGatedDetector
from dear.dataset.rajan_dataset import load_rajan_dataset
from dear.dataset.inpaint_dataset import load_inpaint_dataset
from pipelines.utils import run_inference, compute_test_ap


@hydra.main(config_path="../configs", config_name="rajan_mask_gated", version_base=None)
def main(args):
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(args, resolve=True),
            project='dear', group=args.wandb_group, name=args.wandb_name,
            id=str(uuid.uuid4()), dir=tempfile.mkdtemp(),
        )

    os.makedirs(args.save_path, exist_ok=True)

    detector = RajanMaskGatedDetector(
        lr=args.lr, beta1=args.beta1, weight_decay=args.weight_decay,
        device=args.device, dropout=args.dropout, batched_syncing=args.batched_syncing,
    )

    if args.mode == "train":
        # Start from the pretrained base detector.
        print(f"Loading pretrained base detector from {args.pretrain_path}")
        detector.load(args.pretrain_path)

        # Main training data (aligned real/fake pairs) + diagnostic inpaint data.
        train_dataset, val_dataset = load_rajan_dataset(
            args.data_root, args.train_real_types, args.train_fake_types,
            args.val_real_types, args.val_fake_types, transform_cfg=args.transform,
            use_inversions=args.use_inversions, batched_syncing=args.batched_syncing,
            seed=args.seed,
        )
        inpaint_dataset = load_inpaint_dataset(
            args.data_root, args.train_fake_inpaint_types,
            args.train_fake_inpaint_mask_types, transform_cfg=args.transform,
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
        inpaint_loader = torch.utils.data.DataLoader(
            inpaint_dataset, batch_size=args.inpaint_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        # DISSECT (RAD) + PRUNE (bilateral) + freeze backbone + reset head.
        print("Setting up DEAR gating (dissect + prune)...")
        gating_info = detector.setup_gating(
            inpaint_loader_for_scoring=inpaint_loader,
            exclude_top_ratio=args.exclude_top_ratio,
            exclude_bottom_ratio=args.exclude_bottom_ratio,
            lr=args.lr, beta1=args.beta1, weight_decay=args.weight_decay,
            max_score_batches=args.max_score_batches,
        )
        print(f"Gating info: {gating_info}")
        if args.use_wandb:
            wandb.log(gating_info, step=0)

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

            # ----- Training (head only): CE(main) + CE(inpaint) -----
            detector.train()
            main_iter = iter(train_loader)
            inpaint_iter = iter(inpaint_loader)

            for _ in tqdm.tqdm(range(len(train_loader)), desc=f"Epoch {epoch} [train]"):
                main_batch = next(main_iter)
                try:
                    inpaint_batch = next(inpaint_iter)
                except StopIteration:
                    inpaint_iter = iter(inpaint_loader)
                    inpaint_batch = next(inpaint_iter)

                # Move main batch (paired tuple when batched_syncing) to device.
                if args.batched_syncing:
                    real_batch, fake_batch = main_batch
                    real_batch = {k: v.to(args.device) if torch.is_tensor(v) else v
                                  for k, v in real_batch.items()}
                    fake_batch = {k: v.to(args.device) if torch.is_tensor(v) else v
                                  for k, v in fake_batch.items()}
                    main_batch = (real_batch, fake_batch)
                else:
                    main_batch = {k: v.to(args.device) if torch.is_tensor(v) else v
                                  for k, v in main_batch.items()}

                inpaint_batch = {k: v.to(args.device) if torch.is_tensor(v) else v
                                 for k, v in inpaint_batch.items()}

                loss_dict = detector.combined_update(main_batch, inpaint_batch)
                n_gradient_step += 1
                if n_gradient_step % 1000 == 0 and args.use_wandb:
                    wandb.log(loss_dict, step=n_gradient_step)

            if epoch % args.save_every == 0:
                detector.save(f"{args.save_path}/model_{epoch}.pth")

    elif args.mode == "inference":
        run_inference(detector, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
