"""Check whether the __version__ attribute is set correctly."""

from absl.testing import absltest

import algoperf


class VersionTest(absltest.TestCase):
  def test_version_attribute(self):
    """Check whether __version__ exists and is a valid string."""

    self.assertTrue(hasattr(algoperf, '__version__'))
    version = algoperf.__version__
    self.assertIsInstance(version, str)
    version_elements = version.split('.')
    self.assertTrue(all(el.isnumeric() for el in version_elements))


if __name__ == '__main__':
  absltest.main()
