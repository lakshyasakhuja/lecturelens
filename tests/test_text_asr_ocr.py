"""The parts of text analysis, ASR and OCR that are logic rather than models."""

from __future__ import annotations

from slideseek.asr import (
    AsrSegment,
    Word,
    build_hotwords,
    iter_words,
    jargon_terms,
    load_segments,
    merge_word_pieces,
    save_segments,
    words_between,
)
from slideseek.ocr import OcrLine, SlideText, reading_order
from slideseek.text import analyze, match_spans, tokenize

# ---- text ------------------------------------------------------------------------------


def test_tokenize_splits_on_punctuation_and_lowercases():
    assert tokenize("L2-norm: ||w||^2 (Bellman's)") == ["l2", "norm", "w", "2", "bellman", "s"]


def test_analyze_drops_stopwords_and_stems():
    assert analyze("Why do we need the validation sets?") == ["whi", "need", "valid", "set"]
    assert analyze("validating a set") == ["valid", "set"]
    assert analyze("the of and") == []


def test_analyze_keeps_question_words_and_negation():
    terms = analyze("why is it not converging")
    assert "whi" in terms and "not" in terms


def test_analyze_keeps_digits_but_drops_single_letters():
    assert analyze("k = 5 folds") == ["5", "fold"]
    assert analyze("L2", stemming=False) == ["l2"]


def test_match_spans_highlights_inflected_forms():
    text = "Regularisation penalises large weights; we regularise the model."
    spans = match_spans(text, analyze("regularise weight"))
    assert [text[a:b] for a, b in spans] == ["Regularisation", "weights", "regularise"]
    assert match_spans(text, []) == []


# ---- asr -------------------------------------------------------------------------------


def _segments():
    return [
        AsrSegment(
            0.0, 2.0, "Hello class.", (Word("Hello", 0.1, 0.6, 0.99), Word("class.", 0.6, 1.2, 0.9))
        ),
        AsrSegment(5.0, 7.0, "Bayes' rule.", (Word("Bayes'", 5.0, 5.5), Word("rule.", 5.5, 6.1))),
    ]


def test_segments_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "asr" / "lec01.jsonl"
    save_segments(path, _segments())
    assert load_segments(path) == _segments()
    assert not list(path.parent.glob("*.partial"))


def test_word_pieces_are_merged_into_whole_words():
    pieces = [
        (" This", 0.0, 0.2, 0.9),
        (" held", 0.2, 0.5, 0.8),
        ("-out", 0.5, 0.8, 0.6),  # no leading space: continues "held"
        (" portion", 0.8, 1.2, 0.95),
        (" ", 1.2, 1.2, 0.5),  # empty after stripping: dropped
        (" isn", 1.3, 1.4, 0.9),
        ("'t", 1.4, 1.5, 0.7),
    ]
    words = merge_word_pieces(pieces)
    assert [w.text for w in words] == ["This", "held-out", "portion", "isn't"]
    held_out = words[1]
    assert (held_out.start, held_out.end) == (0.2, 0.8)
    assert held_out.prob == 0.6  # the weaker piece decides
    # A piece with no leading space at the very start has nothing to join.
    assert [w.text for w in merge_word_pieces([("Hello", 0, 1, 1.0)])] == ["Hello"]
    assert merge_word_pieces([]) == ()


def test_iter_words_and_words_between():
    words = list(iter_words(_segments()))
    assert [w.text for w in words] == ["Hello", "class.", "Bayes'", "rule."]
    assert [w.text for w in words_between(words, 0.5, 5.3)] == ["class.", "Bayes'"]


def test_jargon_terms_pick_vocabulary_and_rank_by_slide_frequency():
    slides = [
        "The Bellman equation\nValue of a state under policy pi",
        "Solving the Bellman equation\nUse k-fold validation and the L2 penalty",
        "Summary\nThe softmax function and regularisation",
    ]
    terms = jargon_terms(slides)
    assert terms[0] == "Bellman"  # on two slides, and a proper noun
    assert {"k-fold", "L2", "regularisation", "softmax", "validation"} <= set(terms)
    # Short everyday words and stopwords are not worth prompt tokens.
    assert not {"The", "the", "Use", "and", "state", "Value"} & set(terms)


def test_jargon_terms_respect_the_budget():
    slides = [" ".join(f"Terminology{i:03d}" for i in range(300))]
    terms = jargon_terms(slides, max_terms=50, max_chars=200)
    assert 1 <= len(terms) <= 50
    assert sum(len(term) + 2 for term in terms) <= 200


def test_hotwords_are_phrased_as_a_sentence():
    # A bare comma list makes Whisper drop full stops from the transcript; see build_hotwords.
    assert (
        build_hotwords(["The Bellman equation uses softmax"])
        == "Glossary: Bellman, equation, softmax."
    )
    assert build_hotwords([]) == ""
    assert build_hotwords(["a of the"]) == ""


# ---- ocr -------------------------------------------------------------------------------


def line(text, x0, y0, x1, y1):
    return OcrLine(text, 0.99, (x0, y0, x1, y1))


def test_reading_order_is_top_to_bottom_then_left_to_right():
    lines = [
        line("second bullet", 100, 300, 600, 340),
        line("Title", 60, 50, 400, 110),
        line("first bullet", 100, 200, 600, 240),
    ]
    assert [item.text for item in reading_order(lines)] == [
        "Title",
        "first bullet",
        "second bullet",
    ]


def test_reading_order_joins_boxes_on_one_visual_line():
    # Two boxes on the same line, the right-hand one a couple of pixels higher: sorting by y
    # alone would put "rate" before "learning". They are one line of text, a word gap apart.
    lines = [
        line("rate", 420, 198, 520, 238),
        line("learning", 100, 200, 400, 240),
        line("Next line", 100, 260, 400, 300),
    ]
    assert [item.text for item in reading_order(lines)] == ["learning rate", "Next line"]


def test_split_lines_are_rejoined_without_the_doubled_character():
    # Boxes as PP-OCRv5 actually returned them for three bullets of one slide: the detector
    # split each line, and the padded boxes overlap by about one character.
    lines = [
        line("Gradient descent", 69, 64, 524, 119),
        line("Compute", 121, 196, 297, 239),
        line("e the", 275, 196, 361, 235),
        line("gradient of the loss", 356, 188, 711, 241),
        line("Step in", 116, 275, 257, 329),
        line("the", 248, 279, 333, 321),
        line("opposite direction", 314, 276, 649, 328),
        line("Noisy", 118, 477, 232, 528),
        line("y but cheap updates", 207, 477, 575, 529),
        line("Shuffle t", 119, 564, 270, 607),
        line("the data every epoch", 243, 559, 641, 618),
    ]
    assert [item.text for item in reading_order(lines)] == [
        "Gradient descent",
        "Compute the gradient of the loss",
        "Step in the opposite direction",
        "Noisy but cheap updates",
        "Shuffle the data every epoch",
    ]


def test_joined_line_keeps_the_union_box_and_the_lower_confidence():
    left = OcrLine("Noisy", 0.99, (118, 477, 232, 528))
    right = OcrLine("y but cheap updates", 0.90, (207, 477, 575, 529))
    (joined,) = reading_order([right, left])
    assert joined.box == (118, 477, 575, 529)
    assert joined.confidence == 0.90


def test_a_short_word_is_not_dropped_unless_the_boxes_overlap():
    # "a" follows "data" with a normal word gap: both are real words, nothing was doubled.
    lines = [line("training data", 100, 200, 400, 240), line("a second time", 412, 200, 700, 240)]
    assert [item.text for item in reading_order(lines)] == ["training data a second time"]


def test_reading_order_handles_columns_row_by_row():
    lines = [
        line("right top", 700, 200, 1100, 240),
        line("left top", 100, 202, 500, 242),
        line("right bottom", 700, 300, 1100, 340),
        line("left bottom", 100, 301, 500, 341),
    ]
    assert [item.text for item in reading_order(lines)] == [
        "left top", "right top", "left bottom", "right bottom",
    ]  # fmt: skip


def test_slide_text_round_trip_and_text_property():
    slide = SlideText((line("Title", 0, 0, 10, 10), line("Body", 0, 20, 10, 30)))
    assert slide.text == "Title\nBody"
    assert SlideText.from_dict(slide.to_dict()) == slide
    assert SlideText().text == ""
