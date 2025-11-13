import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from bitrecs.utils.constants import ROOT_DIR


class TestRootDirFromConstants:
    """Test ROOT_DIR imported from bitrecs.utils.constants"""
    
    def test_root_dir_exists_and_is_valid(self):
        """Test that ROOT_DIR from constants exists and is valid"""
        print(f"ROOT_DIR from constants: {ROOT_DIR}")
        
        assert ROOT_DIR.exists(), f"ROOT_DIR does not exist: {ROOT_DIR}"
        assert ROOT_DIR.is_dir(), f"ROOT_DIR is not a directory: {ROOT_DIR}"
        assert ROOT_DIR.is_absolute(), f"ROOT_DIR should be absolute: {ROOT_DIR}"
    
    def test_root_dir_contains_project_files(self):
        """Test that ROOT_DIR contains expected project files"""
        expected_files = [            
            'README.md', 
            'pyproject.toml',
            'setup.py',
            '.gitignore'
        ]
        
        existing_files = []
        for file in expected_files:
            if (ROOT_DIR / file).exists():
                existing_files.append(file)
                print(f"✓ Found: {file}")
            else:
                print(f"✗ Missing: {file}")
        
        assert len(existing_files) > 0, f"ROOT_DIR missing project markers: {ROOT_DIR}"
    
    def test_root_dir_contains_bitrecs_package(self):
        """Test that ROOT_DIR contains the bitrecs package"""
        bitrecs_dir = ROOT_DIR / 'bitrecs'
        print(f"Looking for bitrecs package at: {bitrecs_dir}")
        
        assert bitrecs_dir.exists(), f"bitrecs package not found: {bitrecs_dir}"
        assert bitrecs_dir.is_dir(), f"bitrecs should be a directory: {bitrecs_dir}"
        
        # Check for expected subdirectories
        expected_subdirs = ['base', 'protocol', 'utils', 'api', 'validator', 'miner']
        found_subdirs = []
        
        for subdir in expected_subdirs:
            if (bitrecs_dir / subdir).exists():
                found_subdirs.append(subdir)
                print(f"✓ Found subdir: {subdir}")
            else:
                print(f"✗ Missing subdir: {subdir}")
        
        assert len(found_subdirs) > 0, f"No expected bitrecs subdirectories found: {expected_subdirs}"
    
    def test_root_dir_vs_alternative_methods(self):
        """Compare ROOT_DIR from constants with alternative resolution methods"""
        
        # Method 1: Using this test file's location
        test_file_root = Path(__file__).parent.parent
        
        # Method 2: Using package location
        import bitrecs
        package_root = Path(bitrecs.__file__).parent.parent
        
        # Method 3: Using project markers
        def get_root_via_markers():
            current = Path(__file__).resolve()
            for parent in current.parents:
                markers = ['requirements.txt', 'pyproject.toml', 'setup.py', '.git']
                if any((parent / marker).exists() for marker in markers):
                    return parent
            return current.parent.parent
        
        marker_root = get_root_via_markers()
        
        print(f"Constants ROOT_DIR: {ROOT_DIR}")
        print(f"Test file method: {test_file_root}")
        print(f"Package method:   {package_root}")
        print(f"Marker method:    {marker_root}")
        
        # All methods should point to existing directories
        assert ROOT_DIR.exists(), f"Constants ROOT_DIR doesn't exist: {ROOT_DIR}"
        assert test_file_root.exists(), f"Test file root doesn't exist: {test_file_root}"
        assert package_root.exists(), f"Package root doesn't exist: {package_root}"
        assert marker_root.exists(), f"Marker root doesn't exist: {marker_root}"
        
        # Check if they're the same (they should be)
        print(f"All methods point to same directory: {ROOT_DIR == test_file_root == package_root == marker_root}")


class TestRootDirConsistency:
    """Test ROOT_DIR consistency across different import methods"""
    
    def test_consistent_across_import_methods(self):
        """Test ROOT_DIR is consistent across different import styles"""
        
        # Method 1: Direct import (already done above)
        from bitrecs.utils.constants import ROOT_DIR as root1
        
        # Method 2: Module import
        import bitrecs.utils.constants as const
        root2 = const.ROOT_DIR
        
        # Method 3: Full path import
        import bitrecs.utils.constants
        root3 = bitrecs.utils.constants.ROOT_DIR
        
        print(f"Direct import:     {root1}")
        print(f"Module import:     {root2}")
        print(f"Full path import:  {root3}")
        
        assert root1 == root2 == root3, f"Inconsistent ROOT_DIR across imports: {root1}, {root2}, {root3}"
        assert root1 is root2 is root3, "ROOT_DIR should be the same object (cached)"
    
    def test_root_dir_from_different_working_directories(self):
        """Test ROOT_DIR consistency when changing working directory"""
        original_cwd = os.getcwd()
        original_root = ROOT_DIR
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Original cwd: {original_cwd}")
                print(f"Original ROOT_DIR: {original_root}")
                
                # Change working directory
                os.chdir(temp_dir)
                print(f"Changed cwd to: {os.getcwd()}")
                
                # ROOT_DIR should remain the same (already imported)
                assert ROOT_DIR == original_root, f"ROOT_DIR changed with cwd: {ROOT_DIR} vs {original_root}"
                assert ROOT_DIR.exists(), f"ROOT_DIR should still exist: {ROOT_DIR}"
        
        finally:
            os.chdir(original_cwd)


class TestRootDirUsage:
    """Test practical usage of ROOT_DIR from constants"""
    
    def test_construct_common_paths(self):
        """Test constructing common project paths using ROOT_DIR"""
        
        common_paths = {
            'tests': ROOT_DIR / 'tests',
            'docs': ROOT_DIR / 'docs', 
            'bitrecs': ROOT_DIR / 'bitrecs',
            'requirements': ROOT_DIR / 'pyproject.toml',
            'readme': ROOT_DIR / 'README.md'
        }
        
        print("Testing common path construction:")
        existing_paths = {}
        for name, path in common_paths.items():
            exists = path.exists()
            existing_paths[name] = exists
            print(f"  {name:12}: {path} {'✓' if exists else '✗'}")
        
        # At least some paths should exist
        assert any(existing_paths.values()), f"No common paths found in ROOT_DIR: {ROOT_DIR}"
        
        # bitrecs package should definitely exist
        assert existing_paths['bitrecs'], f"bitrecs package must exist: {common_paths['bitrecs']}"
    
    def test_relative_path_operations(self):
        """Test that ROOT_DIR works for relative path operations"""
        
        # Test path joining
        config_path = ROOT_DIR / '.env'
        log_path = ROOT_DIR / 'logs' / 'app.log'
        data_path = ROOT_DIR / 'data' / 'models'
        
        print(f"Config path: {config_path}")
        print(f"Log path:    {log_path}")
        print(f"Data path:   {data_path}")
        
        # Paths should be under ROOT_DIR
        assert str(ROOT_DIR) in str(config_path), "Config path should be under ROOT_DIR"
        assert str(ROOT_DIR) in str(log_path), "Log path should be under ROOT_DIR"
        assert str(ROOT_DIR) in str(data_path), "Data path should be under ROOT_DIR"
        
        # Paths should be absolute
        assert config_path.is_absolute(), "Config path should be absolute"
        assert log_path.is_absolute(), "Log path should be absolute"  
        assert data_path.is_absolute(), "Data path should be absolute"


class TestRootDirEdgeCases:
    """Test ROOT_DIR behavior in edge cases"""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_root_dir_without_environment_vars(self):
        """Test ROOT_DIR works without environment variables"""
        # Clear environment and test ROOT_DIR still works
        assert ROOT_DIR.exists(), "ROOT_DIR should work without environment variables"
    
    def test_root_dir_string_conversion(self):
        """Test ROOT_DIR string conversion and representation"""
        root_str = str(ROOT_DIR)
        root_repr = repr(ROOT_DIR)
        
        print(f"ROOT_DIR str():  {root_str}")
        print(f"ROOT_DIR repr(): {root_repr}")
        
        assert len(root_str) > 0, "ROOT_DIR string should not be empty"
        assert 'bitrecs' in root_str.lower(), "ROOT_DIR should reference bitrecs project"
        
        # Test platform-appropriate separators
        if os.name == 'nt':  # Windows
            assert '\\' in root_str or '/' in root_str, "Windows path should have separators"
        else:  # Unix-like
            assert '/' in root_str, "Unix path should have forward slashes"
    
    def test_root_dir_parent_operations(self):
        """Test ROOT_DIR parent directory operations"""
        parent = ROOT_DIR.parent
        parents_list = list(ROOT_DIR.parents)
        
        print(f"ROOT_DIR:        {ROOT_DIR}")
        print(f"ROOT_DIR.parent: {parent}")
        print(f"Parents count:   {len(parents_list)}")
        
        assert parent.exists(), f"ROOT_DIR parent should exist: {parent}"
        assert len(parents_list) > 0, "ROOT_DIR should have parent directories"
        assert ROOT_DIR != parent, "ROOT_DIR should not equal its parent"


def test_root_dir_main_functionality():
    """Main test to verify ROOT_DIR from constants works as expected"""
    print(f"\n{'='*60}")
    print("TESTING ROOT_DIR FROM bitrecs.utils.constants")
    print(f"{'='*60}")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"Exists: {ROOT_DIR.exists()}")
    print(f"Is directory: {ROOT_DIR.is_dir()}")
    print(f"Is absolute: {ROOT_DIR.is_absolute()}")
    
    # List contents
    if ROOT_DIR.exists():
        print("\nContents of ROOT_DIR:")
        try:
            for item in sorted(ROOT_DIR.iterdir()):
                item_type = "📁" if item.is_dir() else "📄"
                print(f"  {item_type} {item.name}")
        except PermissionError:
            print("  Permission denied to list contents")
    
    assert ROOT_DIR.exists(), "ROOT_DIR must exist"
    assert ROOT_DIR.is_dir(), "ROOT_DIR must be a directory"


# if __name__ == "__main__":
#     # Run the main test first
#     test_root_dir_main_functionality()
    
#     # Run all tests with pytest
#     pytest.main([__file__, "-v", "-s", "--tb=short"])