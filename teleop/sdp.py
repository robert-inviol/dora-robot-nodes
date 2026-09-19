"""Making a browser's WebRTC offer usable without resolving its mDNS candidate names."""

import ipaddress

CANDIDATE_PREFIX = "a=candidate:"
CANDIDATE_ADDRESS_FIELD = 4
MDNS_SUFFIX = ".local"


def is_usable_candidate_address(address: str) -> bool:
    """A link-local address needs an interface scope, which a candidate line cannot carry."""
    try:
        return not ipaddress.ip_address(address).is_link_local
    except ValueError:
        return False


def reveal_browser_address(sdp: str, browser_address: str) -> str:
    """Replace the mDNS names browsers hide their LAN address behind with the address itself.

    The perception node's ICE stack allows one second to resolve such a name and fails the
    connection when that is not enough. The browser's address is already known here, because
    the offer arrived from it.
    """
    if not is_usable_candidate_address(browser_address):
        return sdp
    return "\r\n".join(_reveal(line, browser_address) for line in sdp.split("\r\n"))


def _reveal(line: str, browser_address: str) -> str:
    if not line.startswith(CANDIDATE_PREFIX):
        return line
    fields = line.split(" ")
    if fields[CANDIDATE_ADDRESS_FIELD].endswith(MDNS_SUFFIX):
        fields[CANDIDATE_ADDRESS_FIELD] = browser_address
    return " ".join(fields)
