The apt repository for Ubuntu (24.04 and 26.04) and Debian (12 and 13), on x86-64 and arm64. The versioned releases hold the same packages; this one is what `apt` reads.

## Install

```sh
sudo install -d /etc/apt/keyrings
curl -fsSL https://github.com/liseman/liberation.al/releases/download/apt/liberational-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/liberational.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/liberational.gpg] https://github.com/liseman/liberation.al/releases/download/apt ./" \
  | sudo tee /etc/apt/sources.list.d/liberational.list
sudo apt update && sudo apt install liberational
pack-setup             # run as yourself, never with sudo: builds your environment and fetches models
```

Then open Liberational from the applications menu, or run `pack-app`.

## Upgrade

```sh
sudo apt update && sudo apt upgrade
```

Check the [release notes](https://github.com/liseman/liberation.al/releases) before upgrading: they say whether a version resets machine-made data. Removing, purging, and pinning a version are covered in [docs/RELEASES.md](https://github.com/liseman/liberation.al/blob/main/docs/RELEASES.md#install-with-apt).
