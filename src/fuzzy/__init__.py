"""Track-A Mamdani Type-1 fuzzy inference system (numpy-only).

Pipeline: concepts -> membership (fuzzify) -> rules (fire) -> fis_score (defuzz) ->
score_matrix, wrapped for RecBole evaluation in run_fuzzy. No scikit-fuzzy.
"""
