//! The one line of facts from the map that goes with an object's view to the describer, and the labels that mark a
//! person.

use world_model::Size;

/// What the map knows of an object besides its view: its size and the labels it was segmented under.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectFacts {
    pub size: Size,
    /// The first reported first.
    pub labels: Vec<String>,
}

fn centimetres(metres: f32) -> u32 {
    (metres * 100.0).round().max(1.0) as u32
}

impl ObjectFacts {
    /// For example `About 12 x 8 x 10 cm (long x short x high). Segmented as: cup, mug.`
    pub fn line(&self) -> String {
        let s = self.size;
        format!(
            "About {} x {} x {} cm (long x short x high). Segmented as: {}.",
            centimetres(s.long_m),
            centimetres(s.short_m),
            centimetres(s.height_m),
            self.labels.join(", ")
        )
    }
}

fn normalised(label: &str) -> String {
    label.split_whitespace().map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}

/// Labels that mark a person, compared ignoring case and spacing. An object reported under any of them is never
/// described or embedded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersonLabels(Vec<String>);

impl PersonLabels {
    pub fn new(labels: &[String]) -> Self {
        Self(labels.iter().map(|l| normalised(l)).collect())
    }

    pub fn any_in<'a>(&self, labels: impl IntoIterator<Item = &'a str>) -> bool {
        labels.into_iter().any(|label| self.0.contains(&normalised(label)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_facts_line_gives_the_size_in_centimetres_and_every_label() {
        let facts = ObjectFacts { size: Size { long_m: 0.123, short_m: 0.08, height_m: 0.1 }, labels: vec!["cup".to_owned(), "mug".to_owned()] };
        assert_eq!(facts.line(), "About 12 x 8 x 10 cm (long x short x high). Segmented as: cup, mug.");
        let tiny = ObjectFacts { size: Size { long_m: 0.002, short_m: 0.001, height_m: 0.0 }, labels: vec!["pin".to_owned()] };
        assert_eq!(tiny.line(), "About 1 x 1 x 1 cm (long x short x high). Segmented as: pin.");
    }

    #[test]
    fn a_person_label_matches_whatever_its_case_and_spacing() {
        let people = PersonLabels::new(&["person".to_owned(), "young child".to_owned()]);
        assert!(people.any_in(["cup", " Person "]));
        assert!(people.any_in(["young   CHILD"]));
        assert!(!people.any_in(["personal computer", "cup"]));
    }
}
