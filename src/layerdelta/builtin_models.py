"""Models of our own that the builtin graph provider can build (registered at import). Test models register themselves."""

from .model import ReferenceCNN
from .providers import register_builtin

register_builtin("reference_cnn", ReferenceCNN, (((1, 1, 28, 28), "float32"),), replace=True)
