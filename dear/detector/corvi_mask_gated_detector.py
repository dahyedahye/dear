import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from dear.nn_classifier.resnet import resnet50


class ChannelGate(nn.Module):
    """Multiply feature-map channels by a fixed gate vector g in [0,1]^C.

    The gate is a (non-trainable) buffer, so it is saved/restored with the
    checkpoint. DEAR uses a hard 0/1 gate to prune channels.
    """

    def __init__(self, num_channels: int, init_value: float = 1.0):
        super().__init__()
        g = torch.full((num_channels,), float(init_value))
        self.register_buffer("gate", g)  # saved in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return x * self.gate.view(1, -1, 1, 1)

    @torch.no_grad()
    def set_gate(self, gate: torch.Tensor):
        gate = gate.detach().to(self.gate.device, dtype=self.gate.dtype)
        gate = gate.clamp_(0.0, 1.0)
        if gate.numel() != self.gate.numel():
            raise ValueError(f"Gate size mismatch: {gate.numel()} vs {self.gate.numel()}")
        self.gate.copy_(gate)

    def ones_(self):
        with torch.no_grad():
            self.gate.fill_(1.0)


class GatedResNet(nn.Module):
    """ResNet-50 whose layer4 features are masked by a channel gate (DEAR prune)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.gate = ChannelGate(num_channels=self.backbone.num_features, init_value=1.0)

    def forward_features_no_gate(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)     # (B, C, h, w)
        feat = self.gate(feat)                       # prune channels
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.backbone.forward_head(feat)


class CorviMaskGatedDetector:
    """DEAR applied to the Corvi base detector (DEAR-c).

    Pipeline (paper Sec. 3):
      1. Load a pretrained base detector.
      2. DISSECT -- score each of the 2048 channels by Regional Activation
         Discrepancy (RAD = mean activation inside the inpainted region minus
         mean activation in the background) over the diagnostic inpaint set.
      3. PRUNE -- bilaterally prune channels at both extremes of the RAD
         distribution, keeping only the robust middle band (a fixed 0/1 gate).
      4. REFINE -- freeze the backbone, re-init the linear head, and retrain it
         on the original training data + the inpaint diagnostic data.

    Only steps 2-4 happen here; the base detector is loaded via ``load()``.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
        pretrained: bool = False,
        dropout: float = 0.0,
    ):
        self.device = device
        self.lr = lr

        backbone = resnet50(pretrained=pretrained, stride0=1, dropout=dropout).change_output(1)
        self.model = GatedResNet(backbone).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        # Optimizer is (re)created in setup_gating(), once the head is reset.
        self.optim = None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path: str):
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt["model"]

        # Loading a raw (un-wrapped) base detector: add the 'backbone.' prefix
        # so weights land inside GatedResNet.backbone. The gate buffer keeps its
        # current value (all-ones), to be set later by setup_gating().
        if not any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return self.loss_fn(pred.squeeze(1), y)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_learning_rate(self):
        if self.optim is None:
            return self.lr
        return self.optim.param_groups[0]['lr']

    def adjust_learning_rate(self, min_lr=1e-6):
        if self.optim is None:
            return False
        for param_group in self.optim.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    # ------------------------------------------------------------------
    # DISSECT: Regional Activation Discrepancy (RAD)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_rad_scores(
        self,
        inpaint_loader,
        max_batches: int | None = None,
        use_apply_align_only: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-channel RAD score, averaged over the diagnostic dataset.

        RAD_k = mean(F_k over inpainted region) - mean(F_k over background).

        High positive RAD -> the channel fires on generated content; strongly
        negative RAD -> it fires on real content; near-zero -> no regional
        preference (these intermediate channels are the robust ones DEAR keeps).
        """
        was_training = self.model.training
        self.model.eval()
        self.model.gate.ones_()  # score on un-gated features

        C = self.model.backbone.num_features
        score_sum = torch.zeros((C,), device=self.device)
        n_used = 0

        total = max_batches if max_batches is not None else len(inpaint_loader)
        pbar = tqdm.tqdm(enumerate(inpaint_loader), total=total, desc="Computing RAD scores")
        for b_idx, batch in pbar:
            if max_batches is not None and b_idx >= max_batches:
                break

            x = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)              # (B,1,H,W)
            apply_align = batch.get("apply_align", None)

            # Keep only samples whose inpaint mask survived the random crop.
            if use_apply_align_only and apply_align is not None:
                apply_align = apply_align.to(self.device).bool()
                if apply_align.sum() == 0:
                    continue
                x = x[apply_align]
                mask = mask[apply_align]

            feat = self.model.forward_features_no_gate(x)     # (B,C,h,w)
            B, C_here, h, w = feat.shape
            assert C_here == C, f"Unexpected channels: {C_here} vs {C}"

            mask_bin = (mask > 0.5).float()
            mask_small = F.interpolate(mask_bin, size=(h, w), mode="bilinear", align_corners=False)
            bg_small = 1.0 - mask_small

            in_mean = (feat * mask_small).sum(dim=(2, 3)) / (mask_small.sum(dim=(2, 3)) + eps)
            bg_mean = (feat * bg_small).sum(dim=(2, 3)) / (bg_small.sum(dim=(2, 3)) + eps)
            score = in_mean - bg_mean                          # (B,C)

            score_sum += score.mean(dim=0)
            n_used += 1

        scores = score_sum / max(n_used, 1)
        if was_training:
            self.model.train()
        return scores.detach().cpu()

    # ------------------------------------------------------------------
    # PRUNE: bilateral channel selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_bilateral_gate(
        self,
        scores: torch.Tensor,
        exclude_top_ratio: float,
        exclude_bottom_ratio: float,
    ) -> torch.Tensor:
        """Keep the middle band of the RAD distribution, prune both extremes.

        m_k = 1[ tau_low <= S_k <= tau_high ], where the top
        ``exclude_top_ratio`` and bottom ``exclude_bottom_ratio`` fractions of
        channels (ranked by RAD) are pruned (gate = 0) and the rest kept
        (gate = 1).
        """
        scores = scores.float()
        C = scores.numel()

        exclude_top_ratio = max(0.0, min(exclude_top_ratio, 1.0))
        exclude_bottom_ratio = max(0.0, min(exclude_bottom_ratio, 1.0))
        if exclude_top_ratio + exclude_bottom_ratio >= 1.0:
            raise ValueError(
                f"exclude_top_ratio ({exclude_top_ratio}) + exclude_bottom_ratio "
                f"({exclude_bottom_ratio}) must be < 1.0 to keep some channels"
            )

        sorted_idx = torch.argsort(scores, descending=True)    # high RAD -> low RAD
        n_top = int(C * exclude_top_ratio)
        n_bot = int(C * exclude_bottom_ratio)
        if C - n_top - n_bot < 1:                              # keep >= 1 channel
            n_top = max(0, n_top - (1 - (C - n_top - n_bot)))

        end_idx = C - n_bot if n_bot > 0 else C
        keep_idx = sorted_idx[n_top:end_idx]

        gate = torch.zeros_like(scores)
        gate[keep_idx] = 1.0
        return gate

    # ------------------------------------------------------------------
    # REFINE: freeze backbone, re-init + retrain the linear head
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _reinit_fc_zero(self):
        fc = self.model.backbone.fc
        fc.weight.zero_()
        if fc.bias is not None:
            fc.bias.zero_()

    def freeze_backbone(self):
        """Freeze the whole backbone except the linear head; keep BN in eval."""
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        for p in self.model.backbone.fc.parameters():
            p.requires_grad = True
        for m in self.model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def setup_gating(
        self,
        inpaint_loader_for_scoring,
        exclude_top_ratio: float,
        exclude_bottom_ratio: float,
        lr: float = 1e-4,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        max_score_batches: int | None = None,
    ):
        """Run DISSECT + PRUNE + freeze, then build the head optimizer."""
        scores = self.compute_rad_scores(
            inpaint_loader_for_scoring, max_batches=max_score_batches,
        )
        print(f"  RAD range: [{scores.min():.4f}, {scores.max():.4f}]")

        gate = self.build_bilateral_gate(scores, exclude_top_ratio, exclude_bottom_ratio)
        self.model.gate.set_gate(gate.to(self.device))

        self.freeze_backbone()
        self._reinit_fc_zero()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=lr, betas=(beta1, 0.999),
                                      weight_decay=weight_decay)

        info = {
            "exclude_top_ratio": exclude_top_ratio,
            "exclude_bottom_ratio": exclude_bottom_ratio,
            "gate_nonzero": int((gate > 0).sum().item()),
            "num_channels": int(gate.numel()),
        }
        return info

    # ------------------------------------------------------------------
    # Training updates (head only; backbone is frozen)
    # ------------------------------------------------------------------
    def update_main(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return self.loss_fn(pred.squeeze(1), y)

    def combined_update(self, main_batch, inpaint_batch):
        """One optimizer step on CE(main) + CE(inpaint).

        The inpaint branch carries a dynamic target (1 if the mask survived the
        crop, else 0), so the refined head learns to flag localized generative
        artifacts as well as whole-image fakes.
        """
        if self.optim is None:
            raise RuntimeError("Optimizer not initialized. Call setup_gating() first.")

        main_loss = self.update_main(main_batch['image'], main_batch['target'].float())
        inpaint_loss = self.update_main(inpaint_batch['image'], inpaint_batch['target'].float())
        total_loss = main_loss + inpaint_loss

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        return {
            "loss": total_loss.item(),
            "main_loss": main_loss.item(),
            "inpaint_ce_loss": inpaint_loss.item(),
        }
