#!/usr/bin/env python3
"""
Result Caching Utilities

Provides save/load functionality for intermediate results to:
- Resume interrupted experiments
- Cache expensive computations
- Share results across experiments
- Version and track experiment outputs

Supports multiple storage formats: pickle, JSON, HDF5, CSV
"""

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
from functools import wraps
import warnings

# Try to import optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class ExperimentCache:
    """
    Cache manager for experiment results with versioning and metadata.
    
    Features:
    - Automatic cache key generation from function arguments
    - Multiple storage backends (pickle, JSON, HDF5)
    - Metadata tracking (timestamps, parameters, versions)
    - Cache invalidation and expiry
    - Resumable experiment checkpoints
    
    Example:
        cache = ExperimentCache('my_experiment')
        
        # Save results
        cache.save('results', {'accuracy': 0.95, 'data': np.array([1,2,3])})
        
        # Load results
        results = cache.load('results')
        
        # Use as decorator
        @cache.cached
        def expensive_computation(param1, param2):
            ...
    """
    
    def __init__(self, experiment_name: str, cache_dir: Optional[Path] = None,
                 default_format: str = 'pickle', version: str = '1.0'):
        """
        Initialize cache manager.
        
        Args:
            experiment_name: Name of the experiment (used for directory)
            cache_dir: Root cache directory (default: experiments/cache)
            default_format: Default storage format ('pickle', 'json', 'hdf5')
            version: Experiment version for cache invalidation
        """
        self.experiment_name = experiment_name
        self.version = version
        self.default_format = default_format
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'cache'
        
        self.cache_dir = Path(cache_dir) / experiment_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.cache_dir / 'metadata.json'
        self._load_metadata()
    
    def _load_metadata(self):
        """Load or initialize metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'experiment_name': self.experiment_name,
                'version': self.version,
                'created': datetime.now().isoformat(),
                'entries': {}
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _get_cache_key(self, key: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key from name and parameters."""
        if params is None:
            return key
        
        # Create hash from parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{key}_{param_hash}"
    
    def _get_filepath(self, cache_key: str, format: str) -> Path:
        """Get file path for cache key."""
        extensions = {
            'pickle': '.pkl',
            'json': '.json',
            'hdf5': '.h5',
            'csv': '.csv',
            'numpy': '.npy'
        }
        ext = extensions.get(format, '.pkl')
        return self.cache_dir / f"{cache_key}{ext}"
    
    def save(self, key: str, data: Any, params: Optional[Dict] = None,
             format: Optional[str] = None, metadata: Optional[Dict] = None) -> Path:
        """
        Save data to cache.
        
        Args:
            key: Cache key name
            data: Data to save (dict, array, DataFrame, etc.)
            params: Parameters used to generate data (for cache key)
            format: Storage format (pickle, json, hdf5, csv, numpy)
            metadata: Additional metadata to store
            
        Returns:
            Path to saved file
        """
        format = format or self.default_format
        cache_key = self._get_cache_key(key, params)
        filepath = self._get_filepath(cache_key, format)
        
        # Save based on format
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self._make_json_serializable(data), f, indent=2)
        
        elif format == 'hdf5':
            if not HAS_H5PY:
                raise ImportError("h5py required for HDF5 format. Install with: pip install h5py")
            self._save_hdf5(filepath, data)
        
        elif format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            elif isinstance(data, dict):
                pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                raise ValueError("CSV format requires DataFrame or dict")
        
        elif format == 'numpy':
            np.save(filepath, data)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Update metadata
        self.metadata['entries'][cache_key] = {
            'key': key,
            'params': params,
            'format': format,
            'filepath': str(filepath),
            'saved_at': datetime.now().isoformat(),
            'size_bytes': filepath.stat().st_size,
            'version': self.version,
            'user_metadata': metadata or {}
        }
        self._save_metadata()
        
        return filepath
    
    def load(self, key: str, params: Optional[Dict] = None,
             format: Optional[str] = None, default: Any = None) -> Any:
        """
        Load data from cache.
        
        Args:
            key: Cache key name
            params: Parameters used when saving
            format: Storage format (auto-detected if not specified)
            default: Default value if cache miss
            
        Returns:
            Cached data or default
        """
        cache_key = self._get_cache_key(key, params)
        
        # Check metadata for format
        if cache_key in self.metadata['entries']:
            entry = self.metadata['entries'][cache_key]
            format = format or entry['format']
            filepath = Path(entry['filepath'])
        else:
            # Try to find file with any extension
            format = format or self.default_format
            filepath = self._get_filepath(cache_key, format)
        
        if not filepath.exists():
            return default
        
        # Load based on format
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        
        elif format == 'hdf5':
            if not HAS_H5PY:
                raise ImportError("h5py required for HDF5 format")
            return self._load_hdf5(filepath)
        
        elif format == 'csv':
            return pd.read_csv(filepath)
        
        elif format == 'numpy':
            return np.load(filepath, allow_pickle=True)
        
        return default
    
    def exists(self, key: str, params: Optional[Dict] = None) -> bool:
        """Check if cache entry exists."""
        cache_key = self._get_cache_key(key, params)
        return cache_key in self.metadata['entries']
    
    def invalidate(self, key: str, params: Optional[Dict] = None):
        """Remove cache entry."""
        cache_key = self._get_cache_key(key, params)
        
        if cache_key in self.metadata['entries']:
            entry = self.metadata['entries'][cache_key]
            filepath = Path(entry['filepath'])
            if filepath.exists():
                filepath.unlink()
            del self.metadata['entries'][cache_key]
            self._save_metadata()
    
    def clear(self):
        """Clear all cached data for this experiment."""
        for cache_key in list(self.metadata['entries'].keys()):
            entry = self.metadata['entries'][cache_key]
            filepath = Path(entry['filepath'])
            if filepath.exists():
                filepath.unlink()
        
        self.metadata['entries'] = {}
        self._save_metadata()
    
    def list_entries(self) -> List[Dict]:
        """List all cache entries with metadata."""
        return [
            {'cache_key': k, **v}
            for k, v in self.metadata['entries'].items()
        ]
    
    def cached(self, key: Optional[str] = None, format: Optional[str] = None):
        """
        Decorator for caching function results.
        
        Args:
            key: Cache key (default: function name)
            format: Storage format
            
        Example:
            @cache.cached()
            def expensive_function(x, y):
                return x + y
        """
        def decorator(func):
            cache_key = key or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create param dict from args/kwargs
                params = {'args': args, 'kwargs': kwargs}
                
                # Check cache
                if self.exists(cache_key, params):
                    return self.load(cache_key, params, format)
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.save(cache_key, result, params, format)
                return result
            
            # Add method to bypass cache
            wrapper.nocache = func
            return wrapper
        
        return decorator
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, np.ndarray):
            return {'__numpy__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
        elif isinstance(obj, pd.DataFrame):
            return {'__dataframe__': True, 'data': obj.to_dict(orient='records')}
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        return obj
    
    def _save_hdf5(self, filepath: Path, data: Dict):
        """Save dictionary to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            self._save_hdf5_recursive(f, data)
    
    def _save_hdf5_recursive(self, group, data: Dict, path: str = ''):
        """Recursively save data to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_hdf5_recursive(subgroup, value, f"{path}/{key}")
            elif isinstance(value, (list, tuple)):
                try:
                    arr = np.array(value)
                    group.create_dataset(key, data=arr)
                except:
                    group.attrs[key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            elif isinstance(value, pd.DataFrame):
                subgroup = group.create_group(key)
                for col in value.columns:
                    subgroup.create_dataset(col, data=value[col].values)
    
    def _load_hdf5(self, filepath: Path) -> Dict:
        """Load dictionary from HDF5 file."""
        result = {}
        with h5py.File(filepath, 'r') as f:
            self._load_hdf5_recursive(f, result)
        return result
    
    def _load_hdf5_recursive(self, group, result: Dict):
        """Recursively load data from HDF5 group."""
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = {}
                self._load_hdf5_recursive(item, result[key])
        
        # Load attributes
        for key, value in group.attrs.items():
            result[key] = value


class CheckpointManager:
    """
    Manage experiment checkpoints for resumable experiments.
    
    Example:
        ckpt = CheckpointManager('long_experiment')
        
        for i in range(100):
            if ckpt.should_skip(i):
                continue
            
            result = run_step(i)
            ckpt.save_checkpoint(i, result)
        
        all_results = ckpt.get_all_results()
    """
    
    def __init__(self, experiment_name: str, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            experiment_name: Name of the experiment
            checkpoint_dir: Directory for checkpoints
        """
        self.experiment_name = experiment_name
        
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
        
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.checkpoint_dir / 'state.json'
        self._load_state()
    
    def _load_state(self):
        """Load checkpoint state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'experiment_name': self.experiment_name,
                'started': datetime.now().isoformat(),
                'completed_steps': [],
                'current_step': None,
                'total_steps': None
            }
            self._save_state()
    
    def _save_state(self):
        """Save checkpoint state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def should_skip(self, step_id: Union[int, str]) -> bool:
        """Check if step was already completed."""
        return str(step_id) in self.state['completed_steps']
    
    def save_checkpoint(self, step_id: Union[int, str], result: Any,
                       metadata: Optional[Dict] = None):
        """
        Save checkpoint for a step.
        
        Args:
            step_id: Step identifier
            result: Result data to save
            metadata: Additional metadata
        """
        step_key = str(step_id)
        
        # Save result
        result_file = self.checkpoint_dir / f"step_{step_key}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump({
                'step_id': step_id,
                'result': result,
                'metadata': metadata,
                'saved_at': datetime.now().isoformat()
            }, f)
        
        # Update state
        if step_key not in self.state['completed_steps']:
            self.state['completed_steps'].append(step_key)
        self.state['current_step'] = step_key
        self.state['last_saved'] = datetime.now().isoformat()
        self._save_state()
    
    def load_checkpoint(self, step_id: Union[int, str]) -> Optional[Any]:
        """Load checkpoint for a step."""
        step_key = str(step_id)
        result_file = self.checkpoint_dir / f"step_{step_key}.pkl"
        
        if result_file.exists():
            with open(result_file, 'rb') as f:
                return pickle.load(f)['result']
        return None
    
    def get_all_results(self) -> Dict:
        """Get all completed results."""
        results = {}
        for step_key in self.state['completed_steps']:
            result_file = self.checkpoint_dir / f"step_{step_key}.pkl"
            if result_file.exists():
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
                    results[data['step_id']] = data['result']
        return results
    
    def get_progress(self) -> Dict:
        """Get progress information."""
        return {
            'completed': len(self.state['completed_steps']),
            'total': self.state.get('total_steps'),
            'current': self.state.get('current_step'),
            'started': self.state.get('started'),
            'last_saved': self.state.get('last_saved')
        }
    
    def set_total_steps(self, total: int):
        """Set total number of steps."""
        self.state['total_steps'] = total
        self._save_state()
    
    def reset(self):
        """Reset all checkpoints."""
        for f in self.checkpoint_dir.glob('step_*.pkl'):
            f.unlink()
        self.state['completed_steps'] = []
        self.state['current_step'] = None
        self._save_state()


# Convenience functions
def cache_result(experiment_name: str):
    """
    Decorator factory for caching function results.
    
    Example:
        @cache_result('my_experiment')
        def compute_metrics(data):
            ...
    """
    cache = ExperimentCache(experiment_name)
    return cache.cached()


def quick_save(data: Any, name: str, experiment: str = 'default') -> Path:
    """Quick save data to cache."""
    cache = ExperimentCache(experiment)
    return cache.save(name, data)


def quick_load(name: str, experiment: str = 'default', default: Any = None) -> Any:
    """Quick load data from cache."""
    cache = ExperimentCache(experiment)
    return cache.load(name, default=default)
