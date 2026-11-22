//! The one rule for whether two detections name the same kind of thing.

use std::fmt;

/// A label as the segmenter wrote it, with the form that labels are compared in.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    text: String,
    key: String,
}

impl Label {
    pub fn new(text: &str) -> Label {
        Label { text: text.to_owned(), key: normalised(text) }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Labels agree when they are the same words, ignoring case and spacing.
    pub fn agrees_with(&self, other: &Label) -> bool {
        self.key == other.key
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// The labels an object's detections carried, each with the sum of the detection scores behind it.
#[derive(Clone, Debug)]
pub struct LabelVotes {
    /// In the order first seen; no two agree.
    votes: Vec<(Label, f32)>,
}

impl LabelVotes {
    pub fn of(label: Label, score: f32) -> LabelVotes {
        LabelVotes { votes: vec![(label, score)] }
    }

    pub fn add(&mut self, label: &Label, score: f32) {
        match self.votes.iter_mut().find(|(voted, _)| voted.agrees_with(label)) {
            Some((_, total)) => *total += score,
            None => self.votes.push((label.clone(), score)),
        }
    }

    pub fn absorb(&mut self, other: LabelVotes) {
        for (label, score) in other.votes {
            self.add(&label, score);
        }
    }

    /// The label with the most score behind it; of equals, the one seen first.
    pub fn winner(&self) -> &Label {
        let mut votes = self.votes.iter();
        let mut winner = votes.next().expect("votes start with one label");
        for vote in votes {
            if vote.1 > winner.1 {
                winner = vote;
            }
        }
        &winner.0
    }
}

fn normalised(text: &str) -> String {
    text.split_whitespace().map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_differing_only_in_case_and_spacing_agree() {
        assert!(Label::new("  Red  Blanket").agrees_with(&Label::new("red blanket")));
    }

    #[test]
    fn the_label_with_the_most_score_behind_it_wins() {
        let mut votes = LabelVotes::of(Label::new("cushion"), 0.9);
        votes.add(&Label::new("pillow"), 0.5);
        assert_eq!(votes.winner().text(), "cushion");
        votes.add(&Label::new("Pillow"), 0.5);
        assert_eq!(votes.winner().text(), "pillow", "two weaker votes outweigh one strong one");
    }

    #[test]
    fn absorbed_votes_add_to_the_labels_they_agree_with() {
        let mut votes = LabelVotes::of(Label::new("cushion"), 0.6);
        let mut other = LabelVotes::of(Label::new("pillow"), 0.5);
        other.add(&Label::new("cushion"), 0.1);
        votes.absorb(other);
        assert_eq!(votes.winner().text(), "cushion");
    }

    #[test]
    fn different_words_do_not_agree() {
        assert!(!Label::new("sofa").agrees_with(&Label::new("sofas")));
        assert!(!Label::new("sofa").agrees_with(&Label::new("cushion")));
    }
}
