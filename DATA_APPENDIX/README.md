# Data Appendix

This directory contains the development records and public evaluation inputs
released with the ATLAS experiments. Directory names follow the terminology
used in the paper.

The Development Set is used for policy construction and calibration and
therefore includes its development reference. The final evaluation sets are
released as public inference inputs only. Evaluation references and hidden-state
assets used by the frozen offline evaluation procedure are maintained separately
and are not included in the public release.

Final-test inference reads only the released input records and the frozen
policy. It does not read evaluation references or hidden-state assets.

## Data Privacy and Provenance

The released files are research benchmark artifacts intended for methodological
evaluation of medication-safety reasoning. Dataset provenance follows the
descriptions provided in the accompanying paper and dataset-specific
documentation.

Benchmark case identifiers are used for research evaluation and should not be
interpreted as clinical medical-record identifiers. The public release is
structured to exclude direct identifying information and private evaluation
assets from the inference-facing benchmark files.

The Development Set reference remains public because the Development Set is
used only for policy construction and calibration. For the final evaluation
sets, public inputs are separated from the private reference or hidden-state
assets used by the frozen offline evaluation procedure.

The released datasets and system outputs are intended for research use only
and are not intended for direct clinical decision-making.

## Integrity

File sizes and SHA-256 checksums for the released public artifacts are listed in
`DATA_APPENDIX_MANIFEST.json`.
