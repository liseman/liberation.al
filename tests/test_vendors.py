"""Device makers and kinds from the IEEE and Bluetooth SIG registries (proximity/vendors.py, #3267). Every registry
here is a small hand-written file in the real files' formats."""

from pathlib import Path

import pytest

from liberational.proximity import vendors

OUI_CSV = """Registry,Assignment,Organization Name,Organization Address
MA-L,F0D5BF,"Acme Radios, Inc.","1 Main St Springfield US 12345 "
MA-L,70B3D5,IEEE Registration Authority,"445 Hoes Lane Piscataway NJ US 08554 "
MA-L,8C1F64,IEEE Registration Authority,"445 Hoes Lane Piscataway NJ US 08554 "
MA-L,02AABB,Nobody Real,"(a locally administered prefix the IEEE would never assign)"
"""
MAM_CSV = """Registry,Assignment,Organization Name,Organization Address
MA-M,8C1F64A,Midblock Sensors GmbH,"Hauptstr. 1 Berlin DE 10115 "
"""
OUI36_CSV = """Registry,Assignment,Organization Name,Organization Address
MA-S,70B3D5123,"Tinyblock Co.,Ltd","Road 2 Taipei TW 100 "
"""
COMPANIES_YAML = """company_identifiers:
  - value: 0x004C
    name: 'Apple, Inc.'
  - value: 0x0006
    name: 'Microsoft'
  - value: 0x0F00
    name: 'O''Brien Audio Ltd'
"""
MEMBERS_YAML = """uuids:
  - uuid: 0xFE07
    name: Sonos, Inc.
"""
SERVICES_YAML = """uuids:
  - uuid: 0x180D
    name: Heart Rate
    id: org.bluetooth.service.heart_rate
  - uuid: 0x180F
    name: Battery
    id: org.bluetooth.service.battery_service
"""


@pytest.fixture
def reg(tmp_path: Path) -> vendors.Registry:
    files = {
        "oui.csv": OUI_CSV,
        "mam.csv": MAM_CSV,
        "oui36.csv": OUI36_CSV,
        "companies.yaml": COMPANIES_YAML,
        "members.yaml": MEMBERS_YAML,
        "services.yaml": SERVICES_YAML,
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return vendors.load(
        [tmp_path / "oui36.csv", tmp_path / "mam.csv", tmp_path / "oui.csv"],
        tmp_path / "companies.yaml",
        tmp_path / "members.yaml",
        tmp_path / "services.yaml",
    )


def ble(identifier: str = "F0:D5:BF:12:34:56", address_type: str | None = "public", **over: object) -> dict:
    """A sighting as proximity/ble.py drains it, with only the fields a lookup reads varied."""
    return {
        "backend": "esp32_serial",
        "identifier_kind": "bluetooth_address",
        "identifier": identifier,
        "address_type": address_type,
        "local_name": None,
        "manufacturer_data": {},
        "service_data": {},
        "service_uuids": [],
        **over,
    }


@pytest.mark.parametrize(
    ("mac", "want"),
    [
        pytest.param("f0:d5:bf:12:34:56", "Acme Radios", id="ma-l-24-bit"),
        pytest.param("8C-1F-64-A1-23-45", "Midblock Sensors", id="ma-m-28-bit"),
        pytest.param("70:B3:D5:12:3F:FF", "Tinyblock", id="ma-s-36-bit"),
        pytest.param("70:B3:D5:12:4F:FF", None, id="unassigned-ma-s-in-an-ieee-block"),
        pytest.param("8C:1F:64:B0:00:00", None, id="unassigned-ma-m-in-an-ieee-block"),
        pytest.param("00:00:00:00:00:01", None, id="unregistered"),
    ],
)
def test_a_network_device_is_named_by_its_longest_registered_block(reg, mac, want):
    assert vendors.network({"mac": mac}, reg) == want


def test_a_locally_administered_network_address_falls_back_to_the_routers_vendor(reg):
    # 0x02 in the first octet: a phone's private Wi-Fi address. Its prefix names no maker, even one in the table.
    assert vendors.network({"mac": "02:AA:BB:00:00:01"}, reg) is None
    assert vendors.network({"mac": "02:AA:BB:00:00:01", "vendor": "Espressif Inc."}, reg) == "Espressif"


def test_the_registry_beats_the_routers_vendor_for_a_public_address(reg):
    assert vendors.network({"mac": "F0:D5:BF:00:00:01", "vendor": "Someone Else"}, reg) == "Acme Radios"


def test_a_public_bluetooth_address_is_named_by_its_block_before_its_payload(reg):
    ad = ble(manufacturer_data={"6": "0109"})  # Microsoft's company id, but the address says Acme
    assert vendors.bluetooth(ad, reg) == ("Acme Radios", None)


@pytest.mark.parametrize(
    "ad",
    [
        pytest.param(ble(address_type="random"), id="random-address"),
        pytest.param(ble(address_type=None), id="unknown-address-type"),
        pytest.param(ble("02:AA:BB:00:00:01"), id="locally-administered"),
        pytest.param(
            ble("6F1B2C3D-0000-4000-8000-000000000001", identifier_kind="corebluetooth_uuid"), id="corebluetooth"
        ),
    ],
)
def test_an_address_that_isnt_a_public_one_names_no_maker(reg, ad):
    assert vendors.bluetooth(ad, reg) == (None, None)


def test_a_random_address_is_named_by_its_manufacturer_data(reg):
    ad = ble("DE:AD:BE:EF:00:2B", "random", manufacturer_data={"3840": "00"})
    assert vendors.bluetooth(ad, reg) == ("O'Brien Audio", None)


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        pytest.param("0719010f2022f58f01", "AirPods or Beats", id="proximity-pairing"),
        pytest.param("1005031c", "iPhone, iPad, Mac, or Watch", id="nearby-info"),
        pytest.param("12025200", "Find My device", id="find-my"),
        pytest.param("ff00", None, id="unknown-type"),
        pytest.param("", None, id="empty"),
    ],
)
def test_apples_continuity_type_says_what_kind_of_device_it_is(reg, payload, kind):
    ad = ble("66:00:00:00:00:95", "random", manufacturer_data={"76": payload})
    assert vendors.bluetooth(ad, reg) == ("Apple", kind)


@pytest.mark.parametrize(
    ("over", "want"),
    [
        pytest.param({"service_uuids": ["0000fe07-0000-1000-8000-00805f9b34fb"]}, ("Sonos", None), id="member-uuid"),
        pytest.param({"service_data": {"fe07": "01"}}, ("Sonos", None), id="member-uuid-in-service-data"),
        pytest.param({"service_uuids": ["0x180D"]}, (None, "Heart Rate"), id="sig-service"),
        pytest.param(
            {"service_uuids": ["0000180f-0000-1000-8000-00805f9b34fb", "0000180d-0000-1000-8000-00805f9b34fb"]},
            (None, "Heart Rate"),
            id="generic-battery-skipped",
        ),
        pytest.param({"service_uuids": ["6e400001-b5a3-f393-e0a9-e50e24dcca9e"]}, (None, None), id="vendor-128-bit"),
    ],
)
def test_advertised_service_uuids_name_a_maker_or_a_kind(reg, over, want):
    assert vendors.bluetooth(ble("DE:AD:BE:EF:00:2B", "random", **over), reg) == want


def test_manufacturer_data_outranks_a_member_uuid(reg):
    ad = ble("DE:AD:BE:EF:00:2B", "random", manufacturer_data={"6": "01"}, service_uuids=["fe07"])
    assert vendors.bluetooth(ad, reg) == ("Microsoft", None)


@pytest.mark.parametrize(
    ("name", "short"),
    [
        ("Apple, Inc.", "Apple"),
        ("Samsung Electronics Co.,Ltd", "Samsung Electronics"),
        ("Espressif Inc.", "Espressif"),
        ("Google LLC", "Google"),
        ("Texas Instruments", "Texas Instruments"),
        ("AG", "AG"),
    ],
)
def test_a_makers_legal_suffixes_are_left_off(name, short):
    assert vendors.short(name) == short


def test_missing_registry_files_mean_no_makers_not_an_error(tmp_path):
    reg = vendors.load([tmp_path / "oui.csv"], tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "c.yaml")
    assert vendors.bluetooth(ble(manufacturer_data={"76": "10"}), reg) == (None, "iPhone, iPad, Mac, or Watch")
    assert vendors.network({"mac": "F0:D5:BF:00:00:01"}, reg) is None
