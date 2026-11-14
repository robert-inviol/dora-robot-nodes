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
    fn different_words_do_not_agree() {
        assert!(!Label::new("sofa").agrees_with(&Label::new("sofas")));
        assert!(!Label::new("sofa").agrees_with(&Label::new("cushion")));
    }
}
