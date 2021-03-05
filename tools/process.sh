# Process data
python tools/compute_softscore.py
python tools/create_dictionary.py
python tools/create_explanation.py --exp qa
python tools/preprocess-hint.py --exp qa