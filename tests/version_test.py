"""Check whether the __version__ attribute is set correctly."""

import algoperf


def test_version_attribute():
  """Check whether __version__ exists and is a valid string."""

  assert hasattr(algoperf, "__version__")
  version = algoperf.__version__
  assert isinstance(version, str)
  version_elements = version.split(".")
  assert all(el.isnumeric() for el in version_elements)
