`pip freeze` of every environment on the machine the artifact was validated on (GCP a3-megagpu-8g,
8x H100 80GB, Ubuntu 22.04, driver 580), written by `env/freeze_envs.sh` on 2026-11-09.
`third_party_commits.txt` lists the upstream commits that were checked out.

`env/setup_envs.sh` installs from the upstream requirement files with the pins it states, which
is what we validated. If a future upstream release breaks one of those installs, recreate the
environment exactly from these files, from the repository root:

```bash
uv venv envs/<name> --python <version on the first line of the file> --python-preference only-managed --seed
uv pip install --python envs/<name>/bin/python -r env/requirements/<name>.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
```

Use `cu126` instead of `cu124` for `freeswim`. For `vbench`, first run `bash env/setup_envs.sh vbench`
once (or clone VBench at the commit in `third_party_commits.txt` and apply the `setup.py` edit
described in `env/setup_envs.sh`), since the file installs `third_party/VBench` in editable mode.
