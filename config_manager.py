from __future__ import annotations
import yaml
from yaml.representer import SafeRepresenter

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, get_type_hints, get_origin, get_args

from config import Config
import logging

CONFIG_VERSION = "1.0"

def represent_tuple(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(tuple, represent_tuple, Dumper=yaml.SafeDumper)

def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return list(dataclass_to_dict(v) for v in obj) # Convert tuple to list for YAML serialization
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj

def dict_to_dataclass(cls, data: Dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    
    # Resolve postponed annotations and forward refs
    try:
        field_types = get_type_hints(cls)
    except Exception:
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    
    constructor_args = {}
    for name, value in data.items():
        if name in field_types:
            field_type = field_types[name]
            origin = get_origin(field_type)
            # Handle Optional/Union
            if origin is not None and origin is not tuple:
                # If Union, find dataclass arg if any
                if origin is list or origin is dict:
                    constructor_args[name] = value
                    continue
                if origin is not None:
                    args = get_args(field_type)
                    dataclass_arg = next((a for a in args if is_dataclass(a)), None)
                    if dataclass_arg and isinstance(value, dict):
                        constructor_args[name] = dict_to_dataclass(dataclass_arg, value)
                        continue
            # Direct dataclass
            if is_dataclass(field_type) and isinstance(value, dict):
                constructor_args[name] = dict_to_dataclass(field_type, value)
                continue
            # Tuple conversion
            if origin is tuple and isinstance(value, list):
                constructor_args[name] = tuple(value)
                continue
            constructor_args[name] = value
        
    return cls(**constructor_args)


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self.active_profile_name: str = "default"
        self.load_or_create()

    def get_active_config(self) -> Config:
        profile_data = self.config_data.get("profiles", {}).get(self.active_profile_name, {})
        return dict_to_dataclass(Config, profile_data)

    def get_profile_config(self, name: str) -> Optional[Config]:
        if name in self.get_profile_names():
            profile_data = self.config_data.get("profiles", {}).get(name, {})
            return dict_to_dataclass(Config, profile_data)
        return None

    def update_active_config(self, config: Config):
        self.config_data.setdefault("profiles", {})[self.active_profile_name] = dataclass_to_dict(config)

    def load_or_create(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config_data = yaml.safe_load(f) or {}
                
                if not self.config_data or "version" not in self.config_data:
                    self._create_default_config()
                
                self.active_profile_name = self.config_data.get("active_profile", "default")

            except (yaml.YAMLError, IOError) as e:
                logging.error(f"Error loading config file: {e}. Creating a new one.")
                self._create_default_config()
        else:
            self._create_default_config()
            self.save()

    def _create_default_config(self):
        default_config_dict = dataclass_to_dict(Config())
        self.config_data = {
            "version": CONFIG_VERSION,
            "active_profile": "default",
            "profiles": {
                "default": default_config_dict
            }
        }
        self.active_profile_name = "default"

    def save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config_data, f, indent=2, Dumper=yaml.SafeDumper)
        except IOError as e:
            print(f"Error saving config file: {e}")

    def get_profile_names(self) -> list[str]:
        return list(self.config_data.get("profiles", {}).keys())

    def set_active_profile(self, name: str):
        if name in self.get_profile_names():
            self.active_profile_name = name
            self.config_data["active_profile"] = name
        else:
            logging.warning(f"Profile '{name}' not found.")

    def create_profile(self, name: str, from_profile: Optional[str] = None):
        if name in self.get_profile_names():
            logging.warning(f"Profile '{name}' already exists.")
            return

        if from_profile and from_profile in self.get_profile_names():
            source_profile = self.config_data["profiles"][from_profile]
            self.config_data["profiles"][name] = deepcopy(source_profile)
        else:
            self.config_data["profiles"][name] = dataclass_to_dict(Config())
        
        self.save()

    def delete_profile(self, name: str):
        if name == "default":
            logging.warning("Cannot delete the default profile.")
            return

        if name in self.get_profile_names():
            del self.config_data["profiles"][name]
            
            if self.active_profile_name == name:
                self.set_active_profile("default")

            self.save()
        else:
            logging.warning(f"Profile '{name}' not found.")
