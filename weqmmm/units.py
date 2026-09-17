"""Conversions between atomic units and OpenMM's nm, kJ/mol, ps convention."""

HARTREE_TO_KJ_MOL = 2625.4996394798254
BOHR_TO_NM = 0.052917721090380
GRAD_AU_TO_FORCE_OMM = HARTREE_TO_KJ_MOL / BOHR_TO_NM
