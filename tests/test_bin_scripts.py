"""Smoke tests for the bin/ CLI scripts (Legacy -> RCS2).

These run each script as a subprocess and check the glue without any network:
  - --help parses and exits 0 (argparse wiring is intact);
  - legacy_query_example writes a valid ra/dec table;
  - the degrade CLIs reach the (blocked) RCS2 sampler contract and fail with a
    clear NotImplementedError -- i.e. everything up to the sampler is wired.

Run as:
    python tests/test_bin_scripts.py
"""

import os
import pathlib
import subprocess
import sys
import tempfile

from astropy.table import Table

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = REPO_ROOT / "bin"
CLI_SCRIPTS = ["query_degrade_legacy", "query_legacy", "read_degrade_legacy",
               "legacy_query_example"]


def _write_props_csv(path):
    """Minimal valid characterize_rcs2-style CSV with grz rows."""
    lines = ["band,frame_id,exp_time,gain,zero_point,seeing,rms,median,n_stars",
             "g,g1,240,1.55,26.40,0.80,18.0,500,30",
             "r,r1,480,1.61,25.95,0.78,31.0,1385,33",
             "z,z1,360,1.55,24.80,0.69,42.0,2130,28"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run(script, *cli_args, cwd=None):
    """Run a bin script in a subprocess with the repo on PYTHONPATH."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, str(BIN / script), *cli_args],
        capture_output=True, text=True, env=env, cwd=cwd,
    )


def test_help_exits_zero():
    """Every argparse CLI responds to --help with exit code 0."""
    for script in CLI_SCRIPTS:
        r = _run(script, "--help")
        assert r.returncode == 0, f"{script} --help rc={r.returncode}\n{r.stderr}"
        assert "usage" in (r.stdout + r.stderr).lower()


def test_example_accepts_csv_input():
    """legacy_query_example reads a .csv coord table and builds the sampler
    (empty table -> completes with no network access)."""
    with tempfile.TemporaryDirectory() as tmp:
        props = os.path.join(tmp, "props.csv")
        _write_props_csv(props)
        coords = os.path.join(tmp, "coords.csv")  # header only, 0 sources
        with open(coords, "w") as f:
            f.write("ra,dec\n")
        r = _run("legacy_query_example", coords,
                 "--rcs2_props_csv", props, "-o", os.path.join(tmp, "out"))
        assert r.returncode == 0, f"rc={r.returncode}\n{r.stderr}"
        assert "Done: 0/0" in r.stdout


def test_query_degrade_builds_sampler_and_runs():
    """query_degrade_legacy builds the bootstrap sampler and runs (empty input
    -> completes with no network access)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "props.csv")
        _write_props_csv(csv)
        table_path = os.path.join(tmp, "in.fits")
        Table({'ra': [], 'dec': []}).write(table_path, overwrite=True)  # 0 rows
        r = _run(
            "query_degrade_legacy", table_path,
            "--rcs2_props_csv", csv, "-o", os.path.join(tmp, "out"),
        )
        assert r.returncode == 0, f"rc={r.returncode}\n{r.stderr}"
        assert "All done" in r.stdout


def test_read_degrade_builds_sampler_and_runs():
    """read_degrade_legacy builds the sampler and runs (empty cache dir)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "props.csv")
        _write_props_csv(csv)
        in_dir = os.path.join(tmp, "cache")
        os.makedirs(in_dir)
        r = _run(
            "read_degrade_legacy", in_dir,
            "--rcs2_props_csv", csv, "-o", os.path.join(tmp, "out"),
        )
        assert r.returncode == 0, f"rc={r.returncode}\n{r.stderr}"
        assert "All done" in r.stdout


def test_missing_csv_fails_fast():
    """A missing --rcs2_props_csv fails before the Pool, with a clear error."""
    with tempfile.TemporaryDirectory() as tmp:
        table_path = os.path.join(tmp, "in.fits")
        Table({'ra': [150.0], 'dec': [2.0]}).write(table_path, overwrite=True)
        r = _run(
            "query_degrade_legacy", table_path,
            "--rcs2_props_csv", os.path.join(tmp, "nope.csv"),
            "-o", os.path.join(tmp, "out"),
        )
        assert r.returncode != 0
        assert "FileNotFoundError" in r.stderr or "No such file" in r.stderr


def _run_all_and_report():
    tests = [
        test_help_exits_zero,
        test_example_accepts_csv_input,
        test_query_degrade_builds_sampler_and_runs,
        test_read_degrade_builds_sampler_and_runs,
        test_missing_csv_fails_fast,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all_and_report())
