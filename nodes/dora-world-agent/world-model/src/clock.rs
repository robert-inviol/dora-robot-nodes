//! Unix times in nanoseconds, and how they are written for the agents: UTC, ISO 8601, to the tenth of a second.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use perception_wire::navigation::UnixNanos;

pub fn now() -> UnixNanos {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).expect("the clock is after 1970");
    UnixNanos(i64::try_from(since_epoch.as_nanos()).expect("the clock is before 2262"))
}

pub fn after(time: UnixNanos, span: Duration) -> UnixNanos {
    UnixNanos(time.0 + i64::try_from(span.as_nanos()).expect("a span shorter than 292 years"))
}

pub fn seconds_between(earlier: UnixNanos, later: UnixNanos) -> f32 {
    ((later.0 - earlier.0) as f64 * 1e-9) as f32
}

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

pub fn iso_time(time: UnixNanos) -> String {
    let tenths = time.0.div_euclid(100_000_000);
    let seconds = tenths.div_euclid(10);
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let in_day = seconds.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{}Z",
        in_day / 3600,
        in_day % 3600 / 60,
        in_day % 60,
        tenths.rem_euclid(10)
    )
}

/// Parses what `iso_time` writes, with or without the tenths.
pub fn parse_iso_time(text: &str) -> Option<UnixNanos> {
    let text = text.strip_suffix('Z')?;
    let (date, time) = text.split_once('T')?;
    let mut date_parts = date.splitn(3, '-').map(str::parse::<i64>);
    let (year, month, day) = (date_parts.next()?.ok()?, date_parts.next()?.ok()?, date_parts.next()?.ok()?);
    let (clock, fraction) = time.split_once('.').unwrap_or((time, "0"));
    let mut clock_parts = clock.splitn(3, ':').map(str::parse::<i64>);
    let (hour, minute, second) = (clock_parts.next()?.ok()?, clock_parts.next()?.ok()?, clock_parts.next()?.ok()?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let fraction_ns = format!("{fraction:0<9}").get(..9)?.parse::<i64>().ok()?;
    // Days from civil, the inverse of civil_from_days.
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let era = shifted_year.div_euclid(400);
    let year_of_era = shifted_year.rem_euclid(400);
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    Some(UnixNanos(((days * 86_400 + hour * 3600 + minute * 60 + second) * 1_000_000_000) + fraction_ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_are_written_in_utc_to_the_tenth_of_a_second() {
        assert_eq!(iso_time(UnixNanos(0)), "1970-01-01T00:00:00.0Z");
        assert_eq!(iso_time(UnixNanos(1_762_560_000_123_456_789)), "2025-11-08T00:00:00.1Z");
        assert_eq!(iso_time(UnixNanos(951_782_400_000_000_000)), "2000-02-29T00:00:00.0Z");
    }

    #[test]
    fn written_times_parse_back() {
        for nanos in [0, 951_782_400_000_000_000, 1_794_610_684_800_000_000] {
            assert_eq!(parse_iso_time(&iso_time(UnixNanos(nanos))), Some(UnixNanos(nanos)));
        }
        assert_eq!(parse_iso_time("2026-11-14T07:00:00Z"), Some(UnixNanos(1_794_639_600_000_000_000)));
        assert_eq!(parse_iso_time("this morning"), None);
        assert_eq!(parse_iso_time("2026-13-01T00:00:00Z"), None);
    }
}
