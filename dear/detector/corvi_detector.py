import torch

from dear.nn_classifier.resnet import resnet50


class CorviDetector:
    """Base Corvi detector: a ResNet-50 binary real/fake classifier.

    Trained on real images (LSUN/COCO) vs. LDM-generated fakes with a single
    logit head and BCE loss. DEAR is applied on top of this pretrained model.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "cpu",
        pretrained: bool = True,
    ):
        self.device = device
        self.lr = lr

        self.model = resnet50(pretrained=pretrained, stride0=1, dropout=0.5).change_output(1)
        self.model.to(self.device)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
            weight_decay=weight_decay,
        )

        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def loss(self, x, y):
        pred_y = self.model(x)
        return self.loss_fn(pred_y.squeeze(1), y)

    def update(self, x, y):
        loss = self.loss(x, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optim.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        return self.optim.param_groups[0]['lr']
