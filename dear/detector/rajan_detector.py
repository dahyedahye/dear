import torch

from dear.detector.corvi_detector import CorviDetector


class RajanDetector(CorviDetector):
    """Base Rajan (AlignedForensics) detector.

    Same ResNet-50 binary classifier as CorviDetector, but supports
    ``batched_syncing`` where each aligned real/fake pair is concatenated and
    updated together.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
        batched_syncing: bool = False,
        pretrained: bool = True,
    ):
        super().__init__(lr, beta1, weight_decay, device, pretrained=pretrained)
        self.batched_syncing = batched_syncing

    def update(self, x, y):
        if self.batched_syncing:
            # x = (real_images, fake_images), y = (real_targets, fake_targets)
            real_images, fake_images = x
            images = torch.cat([real_images, fake_images], dim=0)
            targets = torch.cat([y[0], y[1]], dim=0)

            loss = self.loss(images, targets)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            return {"loss": loss.item()}
        else:
            return super().update(x, y)
