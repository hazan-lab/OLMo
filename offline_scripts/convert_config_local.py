#!/usr/bin/env python3
"""
Script to convert OLMo config from HTTP URLs to local file paths.
"""

import argparse
from pathlib import Path

def convert_config_to_local(input_config: Path, output_config: Path, data_dir: Path):
    """Convert HTTP/S3 URLs in config to local file paths."""
    
    with open(input_config, 'r') as f:
        content = f.read()
    
    # Replace HTTP URLs with local paths
    content = content.replace(
        "http://olmo-data.org/",
        f"{data_dir.absolute()}/"
    )
    
    # Replace HTTPS URLs with local paths
    content = content.replace(
        "https://olmo-data.org/",
        f"{data_dir.absolute()}/"
    )
    
    # Replace S3 URLs with local paths (s3://ai2-llm/ -> data_dir/)
    content = content.replace(
        "s3://ai2-llm/",
        f"{data_dir.absolute()}/"
    )
    
    # Update save folder to local path (if it contains URLs)
    if "http://olmo-data.org/checkpoints" in content:
        content = content.replace(
            "http://olmo-data.org/checkpoints/",
            f"{data_dir.absolute()}/checkpoints/"
        )
    if "https://olmo-data.org/checkpoints" in content:
        content = content.replace(
            "https://olmo-data.org/checkpoints/",
            f"{data_dir.absolute()}/checkpoints/"
        )
    if "s3://ai2-llm/checkpoints" in content:
        content = content.replace(
            "s3://ai2-llm/checkpoints/",
            f"{data_dir.absolute()}/checkpoints/"
        )
    
    with open(output_config, 'w') as f:
        f.write(content)
    
    print(f"Converted config saved to: {output_config}")

def main():
    parser = argparse.ArgumentParser(description="Convert OLMo config to use local data paths")
    parser.add_argument("--input-config", "-i", type=str, default=None,
                       help="Input config file")
    parser.add_argument("--output-config", "-o", type=str, default=None,
                       help="Output config file (only used with --input-config)")
    parser.add_argument("--config-folder", "-f", type=str, default=None,
                       help="Folder containing config files to convert (updates in-place)")
    parser.add_argument("--data-dir", "-d", type=str, default="/scratch/gpfs/EHAZAN/tharuntk/OLMo-data",
                       help="Local data directory")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.config_folder:
        # Process all YAML files in the folder
        config_folder = Path(args.config_folder)
        if not config_folder.exists():
            print(f"Config folder not found: {config_folder}")
            return
        
        yaml_files = list(config_folder.glob("*.yaml")) + list(config_folder.glob("*.yml"))
        
        if not yaml_files:
            print(f"No YAML files found in {config_folder}")
            return
        
        print(f"Found {len(yaml_files)} YAML files in {config_folder}")
        for yaml_file in yaml_files:
            print(f"Converting {yaml_file.name}...")
            convert_config_to_local(yaml_file, yaml_file, data_dir)
        
        print(f"\nConverted {len(yaml_files)} config files in {config_folder}")
        
    elif args.input_config:
        # Process single file
        input_config = Path(args.input_config)
        if not input_config.exists():
            print(f"Input config file not found: {input_config}")
            return
        
        output_config = Path(args.output_config) if args.output_config else input_config
        convert_config_to_local(input_config, output_config, data_dir)
    else:
        parser.error("Either --input-config or --config-folder must be provided")

if __name__ == "__main__":
    main()
