"""
FSE 570 Capstone Project - Earnings Call Intelligence Engine
Phase 0.1: Project Structure Setup
Author: Shalin N. Bhavsar, Ramanuja M. A. Krishna, Freya H. Shah, 
        Kruthika Suresh, Harihara Y. Veldanda
"""

import os
from pathlib import Path

def create_project_structure():
    """
    Creates the complete directory structure for the capstone project.
    """
    
    # Define the project structure
    directories = [
        # Data directories
        'data/raw/transcripts',
        'data/raw/financial',
        'data/raw/market',
        'data/processed/transcripts',
        'data/processed/financial',
        'data/processed/market',
        'data/features',
        'data/final',
        
        # Source code directories
        'src/data_collection',
        'src/preprocessing',
        'src/features',
        'src/models',
        'src/evaluation',
        'src/utils',
        
        # Notebooks
        'notebooks/exploratory',
        'notebooks/modeling',
        'notebooks/results',
        
        # Models
        'models/baseline',
        'models/advanced',
        'models/checkpoints',
        
        # Results
        'results/figures',
        'results/tables',
        'results/reports',
        
        # Logs
        'logs/data_collection',
        'logs/preprocessing',
        'logs/modeling',
        
        # Documentation
        'docs',
        
        # Tests
        'tests',
    ]
    
    # Create directories
    created_count = 0
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_count += 1
            print(f"✅ Created: {directory}")
        else:
            print(f"ℹ️  Already exists: {directory}")
    
    # Create __init__.py files in src subdirectories
    src_dirs = ['src', 'src/data_collection', 'src/preprocessing', 
                'src/features', 'src/models', 'src/evaluation', 'src/utils']
    
    for src_dir in src_dirs:
        init_file = Path(src_dir) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created: {init_file}")
    
    # Create .gitkeep files in data directories to track empty folders
    data_dirs = [d for d in directories if d.startswith('data/')]
    for data_dir in data_dirs:
        gitkeep = Path(data_dir) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    print(f"\n{'='*60}")
    print(f"✅ Project structure created successfully!")
    print(f"📁 Total directories: {len(directories)}")
    print(f"🆕 Newly created: {created_count}")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("FSE 570 CAPSTONE PROJECT - SETUP")
    print("Earnings Call and Risk Intelligence Engine")
    print("="*60)
    print()
    
    create_project_structure()
    
    print("Next steps:")
    print("1. Run: python setup_requirements.py")
    print("2. Run: python verify_installation.py")
    print()