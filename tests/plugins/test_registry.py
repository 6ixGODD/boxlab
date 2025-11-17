from __future__ import annotations

import pathlib

import pytest

from boxlab.dataset.plugins import ExporterPlugin
from boxlab.dataset.plugins import LoaderPlugin
from boxlab.dataset.plugins.registry import get_exporter
from boxlab.dataset.plugins.registry import get_exporter_info
from boxlab.dataset.plugins.registry import get_loader
from boxlab.dataset.plugins.registry import get_loader_info
from boxlab.dataset.plugins.registry import list_exporters
from boxlab.dataset.plugins.registry import list_loaders
from boxlab.dataset.plugins.registry import register_exporter
from boxlab.dataset.plugins.registry import register_loader


class DummyLoader(LoaderPlugin):
    """Dummy loader for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy loader for testing"

    @property
    def supported_extensions(self) -> list[str]:
        return [".dummy"]

    def load(self, _path, **_kwargs):
        from boxlab.dataset import Dataset

        return Dataset(name="dummy_dataset")


class DummyExporter(ExporterPlugin):
    """Dummy exporter for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "Dummy exporter for testing"

    @property
    def default_extension(self) -> str:
        return ".dummy"

    def export(self, dataset, output_dir, **kwargs):
        pass


class TestLoaderRegistry:
    """Test loader registration and retrieval."""

    def test_register_loader(self):
        """Test registering a loader."""
        register_loader("test_loader", DummyLoader)

        # Should be in list
        assert "test_loader" in list_loaders()

    def test_register_duplicate_loader(self):
        """Test that registering duplicate loader raises error."""
        register_loader("test_dup", DummyLoader)

        with pytest.raises(ValueError, match="already registered"):
            register_loader("test_dup", DummyLoader)

    def test_get_loader(self):
        """Test getting a registered loader."""
        register_loader("test_get", DummyLoader)

        loader = get_loader("test_get")
        assert isinstance(loader, DummyLoader)
        assert loader.name == "dummy"

    def test_get_nonexistent_loader(self):
        """Test getting non-existent loader raises error."""
        with pytest.raises(KeyError, match="not found"):
            get_loader("nonexistent_loader")

    def test_list_loaders(self):
        """Test listing all loaders."""
        # Register test loader
        register_loader("test_list", DummyLoader)

        loaders = list_loaders()
        assert isinstance(loaders, list)
        assert "test_list" in loaders

        # Built-in loaders should be registered
        assert "coco" in loaders
        assert "yolo" in loaders

    def test_get_loader_info(self):
        """Test getting loader information."""
        register_loader("test_info", DummyLoader)

        info = get_loader_info()

        assert "test_info" in info
        assert info["test_info"]["name"] == "dummy"
        assert info["test_info"]["description"] == "Dummy loader for testing"
        assert info["test_info"]["supported_extensions"] == [".dummy"]

    def test_loader_instances_are_independent(self):
        """Test that each get_loader call returns a new instance."""
        register_loader("test_instance", DummyLoader)

        loader1 = get_loader("test_instance")
        loader2 = get_loader("test_instance")

        # Should be different instances
        assert loader1 is not loader2
        assert isinstance(loader1, DummyLoader)
        assert isinstance(loader2, DummyLoader)


class TestExporterRegistry:
    """Test exporter registration and retrieval."""

    def test_register_exporter(self):
        """Test registering an exporter."""
        register_exporter("test_exporter", DummyExporter)

        # Should be in list
        assert "test_exporter" in list_exporters()

    def test_register_duplicate_exporter(self):
        """Test that registering duplicate exporter raises error."""
        register_exporter("test_exp_dup", DummyExporter)

        with pytest.raises(ValueError, match="already registered"):
            register_exporter("test_exp_dup", DummyExporter)

    def test_get_exporter(self):
        """Test getting a registered exporter."""
        register_exporter("test_get_exp", DummyExporter)

        exporter = get_exporter("test_get_exp")
        assert isinstance(exporter, DummyExporter)
        assert exporter.name == "dummy"

    def test_get_nonexistent_exporter(self):
        """Test getting non-existent exporter raises error."""
        with pytest.raises(KeyError, match="not found"):
            get_exporter("nonexistent_exporter")

    def test_list_exporters(self):
        """Test listing all exporters."""
        register_exporter("test_list_exp", DummyExporter)

        exporters = list_exporters()
        assert isinstance(exporters, list)
        assert "test_list_exp" in exporters

        # Built-in exporters should be registered
        assert "coco" in exporters
        assert "yolo" in exporters

    def test_get_exporter_info(self):
        """Test getting exporter information."""
        register_exporter("test_info_exp", DummyExporter)

        info = get_exporter_info()

        assert "test_info_exp" in info
        assert info["test_info_exp"]["name"] == "dummy"
        assert info["test_info_exp"]["description"] == "Dummy exporter for testing"

    def test_exporter_instances_are_independent(self):
        """Test that each get_exporter call returns a new instance."""
        register_exporter("test_instance_exp", DummyExporter)

        exporter1 = get_exporter("test_instance_exp")
        exporter2 = get_exporter("test_instance_exp")

        # Should be different instances
        assert exporter1 is not exporter2
        assert isinstance(exporter1, DummyExporter)
        assert isinstance(exporter2, DummyExporter)


class TestBuiltinPlugins:
    """Test that built-in plugins are registered."""

    def test_coco_loader_registered(self):
        """Test that COCO loader is registered."""
        assert "coco" in list_loaders()

        loader = get_loader("coco")
        assert loader.name == "coco"
        assert ".json" in loader.supported_extensions

    def test_yolo_loader_registered(self):
        """Test that YOLO loader is registered."""
        assert "yolo" in list_loaders()

        loader = get_loader("yolo")
        assert loader.name == "yolo"
        assert ".yaml" in loader.supported_extensions or ".yml" in loader.supported_extensions

    def test_coco_exporter_registered(self):
        """Test that COCO exporter is registered."""
        assert "coco" in list_exporters()

        exporter = get_exporter("coco")
        assert exporter.name == "coco"
        assert exporter.default_extension == ".json"

    def test_yolo_exporter_registered(self):
        """Test that YOLO exporter is registered."""
        assert "yolo" in list_exporters()

        exporter = get_exporter("yolo")
        assert exporter.name == "yolo"
        assert exporter.default_extension == ".txt"


class TestRegistryIntegration:
    """Test registry integration with actual plugins."""

    def test_load_dataset_via_registry(self, tmp_coco_dataset: pathlib.Path) -> None:
        """Test loading dataset through registry."""
        loader = get_loader("coco")

        dataset = loader.load(tmp_coco_dataset / "annotations_train.json")

        assert dataset.num_images() == 4
        assert dataset.num_categories() == 2

    def test_export_dataset_via_registry(self, simple_dataset, tmp_path):
        """Test exporting dataset through registry."""
        exporter = get_exporter("coco")

        output_dir = tmp_path / "output"
        exporter.export(simple_dataset, output_dir, copy_images=False)

        assert (output_dir / "annotations_train.json").exists()

    def test_format_auto_detection(self, tmp_coco_dataset, tmp_yolo_dataset):
        """Test auto-detecting format from file extension."""

        def auto_load(path):
            """Auto-detect format and load."""
            path = pathlib.Path(path)

            # Get loader info
            loader_info = get_loader_info()

            # Find matching loader
            for name, info in loader_info.items():
                if path.suffix in info["supported_extensions"]:
                    loader = get_loader(name)
                    return loader.load(path)

            raise ValueError(f"No loader for {path.suffix}")

        # Test COCO
        coco_dataset = auto_load(tmp_coco_dataset / "annotations_train.json")
        assert coco_dataset.num_images() == 4

        # Test YOLO
        yolo_dataset = auto_load(tmp_yolo_dataset / "data.yaml")
        assert yolo_dataset.num_images() == 2

    def test_roundtrip_via_registry(self, tmp_coco_dataset, tmp_path):
        """Test load -> export -> load roundtrip via registry."""
        # Load COCO
        coco_loader = get_loader("coco")
        original = coco_loader.load(tmp_coco_dataset / "annotations_train.json")

        # Export to YOLO
        yolo_exporter = get_exporter("yolo")
        yolo_dir = tmp_path / "yolo"
        yolo_exporter.export(original, yolo_dir, copy_images=True)

        # Load YOLO
        yolo_loader = get_loader("yolo")
        reloaded = yolo_loader.load(yolo_dir / "data.yaml", splits="train")

        # Compare
        assert reloaded.num_images() == original.num_images()
        assert reloaded.num_categories() == original.num_categories()


class TestRegistryErrorHandling:
    """Test error handling in registry."""

    def test_get_loader_with_helpful_error(self):
        """Test that error message lists available loaders."""
        try:
            get_loader("unknown_format")
            raise AssertionError("Should have raised KeyError")
        except KeyError as e:
            error_msg = str(e)
            assert "not found" in error_msg
            assert "Available loaders" in error_msg
            # Should list at least COCO and YOLO
            assert "coco" in error_msg or "yolo" in error_msg

    def test_get_exporter_with_helpful_error(self):
        """Test that error message lists available exporters."""
        try:
            get_exporter("unknown_format")
            raise AssertionError("Should have raised KeyError")
        except KeyError as e:
            error_msg = str(e)
            assert "not found" in error_msg
            assert "Available exporters" in error_msg
