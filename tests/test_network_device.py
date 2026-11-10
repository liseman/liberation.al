from liberational.proximity.network import NetworkDevice


def test_locally_administered_mac_is_not_claimed_randomized():
    row = NetworkDevice("mac_address", "02:00:00:00:00:01", mac="02:00:00:00:00:01", hostname=None, connected=True)
    assert row.locally_administered is True and row.mac_randomized is None
