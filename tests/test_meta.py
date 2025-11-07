from bitrecs.utils.version import LocalMetadata
from bitrecs import __spec_version__ as spec_version
from bitrecs import __version__ as this_version


def test_basic_meta_init_state_ok():
    m = LocalMetadata.local_metadata()
    assert m.head != "head error"
    assert m.remote_head != "remote head error"
    assert m.btversion != "metadata exception"
    assert m.uid == 0
    assert m.coldkey == ""
    assert m.hotkey == ""
    assert m.version == this_version
    assert m.spec == spec_version

    assert isinstance(m.head, str)

    print(m.head)
    print(m.remote_head)
    print(m.btversion)


def test_local_version_matches_head():
    m = LocalMetadata.local_metadata()
    
    if(m.head != m.remote_head):
        print("WARNING Versions do not match - ensure on main branch and updated")

    assert m.head == m.remote_head


def test_version_check_expected_ok():
    m = LocalMetadata.local_metadata()
    match_result = m.version_match()
    if not match_result:
        assert m.head != m.remote_head
    else:
        assert m.head == m.remote_head


def test_version_returns_ok():
    v = LocalMetadata.version()
    assert isinstance(v, str)   
    assert len(v) > 0
    print(f"Version: {v}")
    assert v == this_version
    assert v == "0.8.9"


def test_version_spec_returns_ok():
    s = LocalMetadata.spec()
    assert isinstance(s, str)   
    assert len(s) > 0
    print(f"Spec: {s}")
    