from teleop.sdp import reveal_browser_address

BROWSER = "192.168.111.71"
MDNS_NAME = "ed03282a-d5ce-4d2a-8eae-7f416325ac8b.local"


def _offer(*candidate_lines: str) -> str:
    return "\r\n".join(
        ["v=0", "m=video 9 UDP/TLS/RTP/SAVPF 96", "c=IN IP4 0.0.0.0", *candidate_lines, "a=mid:0", ""]
    )


def test_an_mdns_candidate_gets_the_browsers_address_and_keeps_its_port():
    hidden = f"a=candidate:0 1 UDP 2122252543 {MDNS_NAME} 48314 typ host"

    revealed = reveal_browser_address(_offer(hidden), BROWSER)

    assert f"a=candidate:0 1 UDP 2122252543 {BROWSER} 48314 typ host" in revealed.split("\r\n")
    assert MDNS_NAME not in revealed


def test_every_mdns_candidate_in_the_offer_is_revealed():
    offer = _offer(
        f"a=candidate:0 1 UDP 2122252543 {MDNS_NAME} 48314 typ host",
        f"a=candidate:1 1 TCP 2105524479 {MDNS_NAME} 9 typ host tcptype active",
    )

    assert MDNS_NAME not in reveal_browser_address(offer, BROWSER)


def test_a_candidate_that_already_carries_an_address_is_left_alone():
    offer = _offer("a=candidate:2 1 UDP 1686052863 121.99.161.252 50001 typ srflx raddr 0.0.0.0 rport 0")

    assert reveal_browser_address(offer, BROWSER) == offer


def test_lines_that_are_not_candidates_are_left_alone():
    offer = _offer()

    assert reveal_browser_address(offer, BROWSER) == offer


def test_the_offer_is_left_alone_when_the_browser_came_in_over_a_link_local_address():
    offer = _offer(f"a=candidate:0 1 UDP 2122252543 {MDNS_NAME} 48314 typ host")

    assert reveal_browser_address(offer, "fe80::1c2d:3e4f:5a6b:7c8d") == offer


def test_the_offer_is_left_alone_when_the_browsers_address_is_not_an_ip_address():
    offer = _offer(f"a=candidate:0 1 UDP 2122252543 {MDNS_NAME} 48314 typ host")

    assert reveal_browser_address(offer, "fe80::1%wlan0") == offer
