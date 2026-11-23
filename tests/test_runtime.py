import pytest

from fragtrust.runtime import available_cpus, cgroup_cpu_quota, cpu_threads


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"cpu.max": "1600000 100000\n"}, 16.0),
        ({"cpu.max": "max 100000\n"}, None),
        ({"cpu/cpu.cfs_quota_us": "800000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 8.0),
        ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, None),
        ({"cpu.max": "garbage"}, None),
        ({}, None),
    ],
)
def test_cgroup_cpu_quota(tmp_path, files, expected):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    assert cgroup_cpu_quota(tmp_path) == expected


def test_thread_defaults_stay_within_available_cpus():
    assert 1 <= available_cpus()
    assert cpu_threads() == max(1, available_cpus() - 1)
    assert cpu_threads(3) == 3
