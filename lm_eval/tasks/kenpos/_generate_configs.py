#!/usr/bin/env python3
"""
KenPOS Config Generator

Generates YAML configuration files for all KenPOS languages.
These configs will use generate_until for text generation.

Run: python _generate_configs.py
"""

import yaml


def generate_kenpos_configs():
    """Generate YAML configuration files for all KenPOS language subsets."""
    
    print("=" * 60)
    print("KenPOS Config Generator")
    print("=" * 60)
    print()
    
    # Dataset configuration
    dataset_path = "Kencorpus/KenPOS"
    
    # Language configurations - using actual dataset config names
    # dho = Dholuo, lch = Lumarachi, llg = Lulogooli, lbk = Lubukusu
    languages = {
        "dho": {
            "full_name": "Dholuo",
            "language_family": "Nilotic",
            "approx_words": 50000,
            "description": "Dholuo language"
        },
        "lch": {
            "full_name": "Lumarachi",
            "language_family": "Bantu (Luhya)",
            "approx_words": 27900,
            "description": "Luhya-Lumarachi dialect"
        },
        "llg": {
            "full_name": "Lulogooli",
            "language_family": "Bantu (Luhya)",
            "approx_words": 34300,
            "description": "Luhya-Lulogooli dialect"
        },
        "lbk": {
            "full_name": "Lubukusu",
            "language_family": "Bantu (Luhya)",
            "approx_words": 30900,
            "description": "Luhya-Lubukusu dialect"
        }
    }
    
    print(f"Dataset: {dataset_path}")
    print(f"Output type: generate_until (text generation)")
    print(f"Generating configs for: {list(languages.keys())}")
    print()
    
    # Generate YAML file for each language
    for lang_code, lang_info in languages.items():
        print(f"Generating config for {lang_code} ({lang_info['full_name']})...")
        
        # Create YAML configuration
        config = {
            "include": "_default_template_yaml",
            "task": f"kenpos_{lang_code}",
            "dataset_name": lang_code,
            "metadata": {
                "version": 1.0,
                "language": lang_code,
                "full_name": lang_info["full_name"],
                "language_family": lang_info["language_family"],
                "approx_words": lang_info["approx_words"],
                "description": f"Part-of-Speech tagging for {lang_info['description']} (~{lang_info['approx_words']:,} words)"
            }
        }
        
        # Write YAML file
        yaml_filename = f"{lang_code}.yaml"
        with open(yaml_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Created: {yaml_filename}")
    
    print()
    print("=" * 60)
    print("Config generation complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    for lang_code in languages.keys():
        print(f"  - {lang_code}.yaml")
    print()
    print("Task names (use these with --tasks):")
    for lang_code in languages.keys():
        print(f"  - kenpos_{lang_code}")
    print()
    print("Next steps:")
    print("  1. Verify files: ls *.yaml")
    print("  2. Check tasks: lm_eval --tasks list | grep kenpos")
    print("  3. Quick test: lm_eval --model hf --model_args pretrained=gpt2 --tasks kenpos_dho --limit 5")
    print("  4. Full eval: lm_eval --model hf --model_args pretrained=your-model --tasks kenpos --device cuda:0")
    print()


if __name__ == "__main__":
    # Check for yaml package
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not installed")
        print("Install with: pip install pyyaml")
        exit(1)
    
    generate_kenpos_configs()