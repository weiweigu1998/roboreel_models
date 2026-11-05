# Vendored `openpi`

This directory is a **partial vendored copy** of the `openpi` package from
Physical Intelligence: https://github.com/Physical-Intelligence/openpi

Only the importable package (upstream `src/openpi/`) is copied, not the whole
repository, so it is vendored rather than used as a git submodule — the
`demonstration_vla_suite` Pi0.5 path imports it as `openpi.*`.

## Licensing — ACTION REQUIRED BEFORE PUBLIC RELEASE

Upstream is **Apache-2.0** and additionally ships `LICENSE_GEMMA.txt`, because
part of the tree is derived from Google's Gemma. This vendored copy includes
those Gemma-derived files:

- `models/gemma.py`, `models/gemma_fast.py`
- `models_pytorch/gemma_pytorch.py`
- `models_pytorch/transformers_replace/models/gemma/{configuration_gemma,modeling_gemma}.py`
- `models_pytorch/transformers_replace/models/paligemma/*`

Before publishing, this directory must carry **both**:

1. `LICENSE` — the upstream Apache-2.0 license text, with the upstream
   copyright notice intact.
2. `LICENSE_GEMMA.txt` — the Gemma Terms of Use, as upstream provides it.

Record the exact upstream commit the copy was taken from here, so downstream
users can diff against it:

    upstream commit: <TODO: fill in>

Local modifications to the vendored tree, if any, should be listed below.

    modifications: <TODO: none / list them>
