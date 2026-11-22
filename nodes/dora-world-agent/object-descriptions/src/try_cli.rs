//! The arguments of the `describe_and_rank` example: one view, its facts, and the queries to rank against it.

use std::path::PathBuf;

use world_model::Size;

pub const USAGE: &str = "usage: describe_and_rank --settings <spatial-agent.toml> --image <view.jpg> --size <long>x<short>x<height> \
(centimetres) --labels <label>[,<label>...] <query> [<query>...]";

/// What one run describes and ranks.
#[derive(Clone, Debug, PartialEq)]
pub struct TryArguments {
    /// The spatial agent's settings file; its `[claude]` and `[object_descriptions]` sections are used.
    pub settings: PathBuf,
    /// A JPEG of the object, as the object map cuts its views.
    pub image: PathBuf,
    pub size: Size,
    pub labels: Vec<String>,
    pub queries: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BadArguments {
    #[error("{0} needs a value")]
    MissingValue(&'static str),
    #[error("{0} is missing")]
    Missing(&'static str),
    #[error("{0} is given twice")]
    Repeated(&'static str),
    #[error("{0:?} is not a size like 12x8x10 in centimetres")]
    BadSize(String),
    #[error("--labels names no label")]
    NoLabels,
    #[error("no query to rank")]
    NoQueries,
}

fn parse_size(text: &str) -> Result<Size, BadArguments> {
    let centimetres: Vec<f32> = text
        .split('x')
        .map(|part| part.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0))
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| BadArguments::BadSize(text.to_owned()))?;
    match centimetres[..] {
        [long, short, height] => Ok(Size { long_m: long / 100.0, short_m: short / 100.0, height_m: height / 100.0 }),
        _ => Err(BadArguments::BadSize(text.to_owned())),
    }
}

fn set<T>(slot: &mut Option<T>, flag: &'static str, value: T) -> Result<(), BadArguments> {
    match slot.replace(value) {
        Some(_) => Err(BadArguments::Repeated(flag)),
        None => Ok(()),
    }
}

impl TryArguments {
    /// The arguments after the program name.
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, BadArguments> {
        let (mut settings, mut image, mut size, mut labels) = (None, None, None, None);
        let mut queries = Vec::new();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let flag: &'static str = match argument.as_str() {
                "--settings" => "--settings",
                "--image" => "--image",
                "--size" => "--size",
                "--labels" => "--labels",
                _ => {
                    queries.push(argument);
                    continue;
                }
            };
            let value = arguments.next().ok_or(BadArguments::MissingValue(flag))?;
            match flag {
                "--settings" => set(&mut settings, flag, PathBuf::from(value))?,
                "--image" => set(&mut image, flag, PathBuf::from(value))?,
                "--size" => set(&mut size, flag, parse_size(&value)?)?,
                _ => {
                    let named: Vec<String> = value.split(',').map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect();
                    if named.is_empty() {
                        return Err(BadArguments::NoLabels);
                    }
                    set(&mut labels, flag, named)?;
                }
            }
        }
        if queries.is_empty() {
            return Err(BadArguments::NoQueries);
        }
        Ok(Self {
            settings: settings.ok_or(BadArguments::Missing("--settings"))?,
            image: image.ok_or(BadArguments::Missing("--image"))?,
            size: size.ok_or(BadArguments::Missing("--size"))?,
            labels: labels.ok_or(BadArguments::Missing("--labels"))?,
            queries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &[&str]) -> Result<TryArguments, BadArguments> {
        TryArguments::parse(text.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn every_flag_and_the_queries_after_them_are_read() {
        let arguments = parse(&[
            "--settings", "agent.toml", "--image", "cup.jpg", "--size", "12x8x10", "--labels", "cup, mug", "red cup", "blue mug",
        ])
        .unwrap();
        assert_eq!(arguments.settings, PathBuf::from("agent.toml"));
        assert_eq!(arguments.image, PathBuf::from("cup.jpg"));
        assert_eq!(arguments.size, Size { long_m: 0.12, short_m: 0.08, height_m: 0.1 });
        assert_eq!(arguments.labels, ["cup", "mug"]);
        assert_eq!(arguments.queries, ["red cup", "blue mug"]);
    }

    #[test]
    fn queries_may_come_before_the_flags() {
        let arguments = parse(&["red cup", "--labels", "cup", "--size", "1x1x1", "--image", "a.jpg", "--settings", "s.toml"]).unwrap();
        assert_eq!(arguments.queries, ["red cup"]);
    }

    #[test]
    fn a_missing_or_malformed_argument_is_named() {
        assert_eq!(parse(&["--image", "a.jpg", "--size", "1x1x1", "--labels", "cup", "q"]), Err(BadArguments::Missing("--settings")));
        assert_eq!(parse(&["--settings", "s", "--image", "a.jpg", "--size", "1x1x1", "--labels", "cup"]), Err(BadArguments::NoQueries));
        assert_eq!(parse(&["q", "--size"]), Err(BadArguments::MissingValue("--size")));
        assert_eq!(parse(&["q", "--size", "12x8"]), Err(BadArguments::BadSize("12x8".to_owned())));
        assert_eq!(parse(&["q", "--size", "12x-8x3"]), Err(BadArguments::BadSize("12x-8x3".to_owned())));
        assert_eq!(parse(&["q", "--labels", " , "]), Err(BadArguments::NoLabels));
        assert_eq!(parse(&["q", "--image", "a", "--image", "b"]), Err(BadArguments::Repeated("--image")));
    }
}
