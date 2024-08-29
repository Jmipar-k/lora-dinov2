import torch
import torch.nn as nn
import torch.nn.init as init


class LinearClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        out_dim: int = 35,
        n_classes: int = 4,
        layers: int = 4,
        pretrained: bool = False,
    ):
        """The Linear Classification Decoder

        Args:
            embed_dim (int): The dimension of the embedding input.
            out_dim (int, optional): The resolution of patch size, essentially
                (img_size / patch_size) of the encoder. Defaults to 490 / 14 = 35.
            n_classes (int, optional): Number of output classes. Defaults to 4.
            layers (int, optional): The number of layers or additional features.
            pretrained (bool, optional): Load pretrained weights if available. Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_classes = n_classes
        self.layers = layers
        self.pretrained = pretrained # TODO: load pretrained weights
        self.classifier_head = nn.Linear(self.out_dim * self.out_dim * self.embed_dim, self.n_classes)
        self.init_weights()

        if self.pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        # Placeholder for loading pretrained weights logic
        pass

    def init_weights(self):
        init.xavier_uniform_(self.classifier_head.weight)
        if self.classifier_head.bias is not None:
            init.constant_(self.classifier_head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.out_dim * self.out_dim * self.embed_dim)
        return self.classifier_head(x)
