from sklearn.base import TransformerMixin, BaseEstimator
import importlib
import os
from bob.extension.download import get_file


class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using Pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.device = device

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        if self.model is None:
            self._load_model()

            self.model.eval()

            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

        return self.model(X)  # .detach().numpy()

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def to(self, device):
        self.device = device

        if self.model is not None:
            self.model.to(self.device)

    def get_model(self):
        if self.model is None:
            self._load_model()
            self.model.eval()
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
        return self.model


def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )


class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self, device="cpu"):
        self.device = device
        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(checkpoint_path, config, device=device)

    def _load_model(self):
        model = load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model


def load_source(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)

    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module

    loader.exec_module(module)
    return module


def build_arcface_model(device="cpu"):
    return IResnet100(device=device)
