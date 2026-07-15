import torch

from dear.detector.corvi_mask_gated_detector import CorviMaskGatedDetector


class RajanMaskGatedDetector(CorviMaskGatedDetector):
    """DEAR applied to the Rajan base detector (DEAR-r).

    Identical DISSECT / PRUNE / REFINE procedure as DEAR-c, but the main
    training branch uses Rajan's aligned real/fake pairs. With
    ``batched_syncing`` the pair is concatenated into a single batch so the
    only difference between the real and fake samples is the generative
    artifact.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
        pretrained: bool = False,
        dropout: float = 0.0,
        batched_syncing: bool = False,
    ):
        super().__init__(
            lr=lr, beta1=beta1, weight_decay=weight_decay,
            device=device, pretrained=pretrained, dropout=dropout,
        )
        self.batched_syncing = batched_syncing

    def combined_update(self, main_batch, inpaint_batch):
        """One optimizer step on CE(main) + CE(inpaint), with paired main batch."""
        if self.optim is None:
            raise RuntimeError("Optimizer not initialized. Call setup_gating() first.")

        if self.batched_syncing:
            # main_batch is a tuple: (real_batch, fake_batch)
            real_batch, fake_batch = main_batch
            x_main = torch.cat([real_batch['image'], fake_batch['image']], dim=0)
            y_main = torch.cat([
                torch.zeros(real_batch['image'].size(0), device=self.device),
                torch.ones(fake_batch['image'].size(0), device=self.device),
            ], dim=0)
        else:
            x_main = main_batch['image']
            y_main = main_batch['target'].float()

        main_loss = self.update_main(x_main, y_main)
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
