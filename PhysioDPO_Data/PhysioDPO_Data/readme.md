output/
├── scored_sequences.jsonl          # Scored sequence data
├── preference_dataset.jsonl        # Preference-pair dataset for DPO training
├── pdbs/                           # Protein structure files for the initial sequences (PDB format)
│   ├── [sequence_id].pdb          # 3D structure file for each sequence
├── pdbs_mutants/                   # Protein structure files for mutant sequences
│   ├── [sequence_id]_mut_[n].pdb  # 3D structure file for each mutant
└── visualizations/                 # Visualization output directory (if generated)

python display.py
python -m http.server 8000