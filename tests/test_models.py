"""Minimal tests for data models."""

import unittest

from src.models import SiteConfig


class TestSiteConfig(unittest.TestCase):
    """Test SiteConfig model."""

    def test_site_id_basic(self):
        site = SiteConfig(url="https://example.com")
        self.assertEqual(site.site_id, "example.com")

    def test_site_id_strips_www(self):
        site = SiteConfig(url="https://www.example.com/about")
        self.assertEqual(site.site_id, "example.com")

    def test_site_id_preserves_subdomain(self):
        site = SiteConfig(url="https://careers.example.com")
        self.assertEqual(site.site_id, "careers.example.com")

    def test_site_id_lowercase(self):
        site = SiteConfig(url="https://WWW.Example.COM")
        self.assertEqual(site.site_id, "example.com")

    def test_name_optional(self):
        site = SiteConfig(url="https://example.com")
        self.assertIsNone(site.name)
        site_named = SiteConfig(url="https://example.com", name="Test")
        self.assertEqual(site_named.name, "Test")


if __name__ == "__main__":
    unittest.main()
