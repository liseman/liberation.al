//! Which browser origins may open a session. A page from any other site could otherwise open a WebSocket to the
//! gateway from the viewer's browser (cross-site WebSocket hijacking: the browser sends no CORS preflight for an
//! upgrade) and read the cameras. The rule: a request with no `Origin` header (a non-browser client, a test) is
//! allowed; a browser's origin is allowed when its host is the host the request was addressed to (the dashboard and
//! the gateway live on the same machine, so the dashboard's page always has the gateway's host, whatever name or
//! address it was reached by) or is on the `--allow-origin` list. Origins are compared as `scheme://host[:port]`,
//! lower-case; the same-host rule ignores the port and scheme (the dashboard is :8765 over http, Vite :5173).
//!
//! The same-host rule alone would fall to DNS rebinding: a page at `evil.example` whose name the attacker points at
//! this machine sends `Origin: http://evil.example` and `Host: evil.example`, and they match. So the request's
//! `Host` must itself be one of this gateway's names: an IP literal (a browser only sends one when the page was
//! opened by address, which no rebinding can forge), or a name on the `hosts` list (the machine's host name, its
//! `.local` name, `localhost`, and anything given with `--allow-host`). A `Host` that is none of those is refused
//! whenever an `Origin` is present.

/// What the gateway allows besides the same host, and which host names it answers to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OriginPolicy {
    /// Extra origins, exactly as a browser sends them (`https://dash.example:8765`).
    pub allowed: Vec<String>,
    /// Host names this gateway is served under, lower-case. A request's `Host` must be one of them or an IP
    /// literal for its `Origin` to be compared at all.
    pub hosts: Vec<String>,
}

impl OriginPolicy {
    /// A policy allowing `allowed` besides the same host, served under the names `hosts`.
    #[must_use]
    pub fn new(allowed: impl IntoIterator<Item = String>, hosts: impl IntoIterator<Item = String>) -> Self {
        Self {
            allowed: allowed.into_iter().map(|o| o.trim_end_matches('/').to_ascii_lowercase()).collect(),
            hosts: hosts
                .into_iter()
                .map(|h| if h.parse::<std::net::Ipv6Addr>().is_ok() { format!("[{h}]") } else { h })
                .filter_map(|h| host_part(&h))
                .collect(),
        }
    }

    /// Whether a request with this `Origin` header (if any), addressed to `host` (the `Host` header, with or
    /// without a port), may open a session.
    #[must_use]
    pub fn allows(&self, origin: Option<&str>, host: Option<&str>) -> bool {
        let Some(origin) = origin else { return true };
        let origin = origin.trim_end_matches('/').to_ascii_lowercase();
        if origin == "null" {
            return false;
        }
        if self.allowed.contains(&origin) {
            return true;
        }
        let Some(h) = host.and_then(host_part) else { return false };
        if !(is_ip_literal(&h) || self.hosts.contains(&h)) {
            return false;
        }
        host_of(&origin).is_some_and(|o| o == h)
    }
}

/// Whether a host is an IP address literal (`127.0.0.1`, `[::1]`): a browser sends one only for a page opened by
/// address, which DNS rebinding can't produce.
fn is_ip_literal(host: &str) -> bool {
    if let Some(inner) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        return inner.parse::<std::net::Ipv6Addr>().is_ok();
    }
    host.parse::<std::net::Ipv4Addr>().is_ok()
}

/// The host of an origin (`https://[::1]:8765` gives `[::1]`), or `None` when it has no scheme.
fn host_of(origin: &str) -> Option<String> {
    let rest = origin.split_once("://")?.1;
    host_part(rest)
}

/// The host of `host[:port]`, bracketed IPv6 kept whole.
fn host_part(authority: &str) -> Option<String> {
    let authority = authority.split('/').next()?;
    if authority.is_empty() {
        return None;
    }
    if let Some(end) = authority.find(']') {
        return authority.starts_with('[').then(|| authority[..=end].to_ascii_lowercase());
    }
    Some(authority.rsplit_once(':').map_or(authority, |(h, _)| h).to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_host_and_listed_origins_pass_and_others_are_refused() {
        let p =
            OriginPolicy::new(["https://Dash.Example/".to_owned()], ["DevBox".to_owned(), "devbox.local:9".to_owned()]);
        assert!(p.allows(None, Some("devbox:8766")), "a non-browser client has no origin");
        assert!(p.allows(Some("http://devbox:8765"), Some("devbox:8766")), "the dashboard on the same host");
        assert!(p.allows(Some("http://DEVBOX:5173"), Some("devbox:8766")), "Vite on the same host, any case");
        assert!(p.allows(Some("http://100.64.0.9:8765"), Some("100.64.0.9:8766")), "by address");
        assert!(p.allows(Some("http://[::1]:8765"), Some("[::1]:8766")), "IPv6");
        assert!(p.allows(Some("https://dash.example"), Some("devbox:8766")), "listed");
        assert!(!p.allows(Some("http://evil.example"), Some("devbox:8766")));
        assert!(!p.allows(Some("http://devbox.evil.example"), Some("devbox:8766")));
        assert!(!p.allows(Some("null"), Some("devbox:8766")), "a sandboxed or file page");
        assert!(!p.allows(Some("http://devbox:8765"), None), "no host to compare with");
        assert!(!p.allows(Some("devbox"), Some("devbox:8766")), "not an origin");
        assert!(p.allows(Some("http://devbox.local:8765"), Some("devbox.local:8766")), "a listed name");
    }

    #[test]
    fn a_rebound_name_is_refused_because_the_host_is_not_one_of_ours() {
        let p = OriginPolicy::new([], ["devbox".to_owned()]);
        // The attacker's page: its own name points at this machine, so Origin and Host agree, but the name isn't ours.
        assert!(!p.allows(Some("http://evil.example"), Some("evil.example:8766")));
        assert!(!p.allows(Some("http://evil.example:8765"), Some("evil.example")));
        // Our names and any address are fine.
        assert!(p.allows(Some("http://devbox:8765"), Some("devbox:8766")));
        assert!(p.allows(Some("http://100.64.0.9:8765"), Some("100.64.0.9:8766")));
        assert!(p.allows(Some("http://[fd7a::1]:8765"), Some("[fd7a::1]:8766")));
        assert!(!p.allows(Some("http://1.2.3.4.5:8765"), Some("1.2.3.4.5:8766")), "not an address, not a name of ours");
        // A listed origin needs no host check (it's the page's own, fixed name).
        let q = OriginPolicy::new(["https://dash.example".to_owned()], []);
        assert!(q.allows(Some("https://dash.example"), Some("evil.example")));
        // With no names at all, only addresses pass.
        assert!(!OriginPolicy::default().allows(Some("http://localhost:5173"), Some("localhost:8766")));
        // An unbracketed IPv6 entry and an empty one are taken in their stride; an upper-case Host matches.
        let r = OriginPolicy::new([], ["fd7a::1".to_owned(), String::new(), "Devbox.Example.ts.net".to_owned()]);
        assert_eq!(r.hosts, vec!["[fd7a::1]".to_owned(), "devbox.example.ts.net".to_owned()]);
        assert!(r.allows(Some("https://devbox.example.ts.net:8443"), Some("DEVBOX.EXAMPLE.TS.NET:8766")));
    }
}
