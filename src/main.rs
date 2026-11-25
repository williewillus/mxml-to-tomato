//! Convert a MusicXML file to a Tomato-Jianpu v1 (`.jps`) script.
//!
//! Supported input (the subset of MusicXML used by single-voice melody parts
//! such as `bangdi_cleaned.musicxml`):
//! - one part, one voice, no lyrics, no grace notes, no tuplets, no repeats
//! - key and time signature changes, multiple-rest blocks (blocks longer
//!   than `COLLAPSE_MULTIREST_ABOVE` collapse to one annotated measure)
//! - slurs and ties, rendered as jianpu slur parentheses `( )`
//! - chord notes (dyads): rendered with the `{dsb ...}` overlay for isolated
//!   chords, or as a second staff (`Q1:`/`Q2:`) for chord runs
//! - dynamics, wedges, accents, tenutos, staccatos, trills, mordents, fermatas
//!
//! Unsupported constructs are reported on stderr and skipped.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;

use musicxml::datatypes::{
    BarStyle, RightLeftMiddle, StartStop, StartStopContinue, Step, WedgeType,
};
use musicxml::elements::{
    ArticulationsType, AudibleType, DirectionTypeContents, DynamicsType, KeyContents,
    MeasureElement, MeasureStyleContents, NotationContentTypes, NoteType, OrnamentType,
    PartElement, PartListElement, ScorePartwise,
};
use musicxml::read_score_partwise;

/// Measures grouped into one printed line (one `Q:` row) of the output.
/// MusicXML carries no line layout for this score, so this is a fixed choice.
const MEASURES_PER_LINE: usize = 4;
/// A run of at least this many consecutive chorded notes becomes a second staff.
const CHORD_RUN_FOR_TWO_STAVES: usize = 3;
/// A line with at least this many chorded notes becomes two staves.
const CHORD_COUNT_FOR_TWO_STAVES: usize = 4;
/// Multi-rest blocks spanning more than this many measures collapse to a
/// single annotated measure (`0"N小節" 0 0 0`, the example script's style).
/// Shorter blocks stay expanded, without the count annotation.
const COLLAPSE_MULTIREST_ABOVE: u32 = 3;

const STEP_LETTERS: [char; 7] = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
/// Order of sharps (F C G D A E B) as step indices (C = 0 ... B = 6).
const SHARP_ORDER: [usize; 7] = [3, 0, 4, 1, 5, 2, 6];
/// Order of flats (B E A D G C F) as step indices.
const FLAT_ORDER: [usize; 7] = [6, 2, 5, 1, 4, 0, 3];

/// A pitch as written: diatonic step (0 = C ... 6 = B), alteration in
/// semitones, and the MusicXML octave (4 = the octave started by middle C).
#[derive(Clone, Copy)]
struct Pitch {
    step: usize,
    alter: i16,
    octave: u8,
}

/// A chord tone attached to a melody note, with its own duration.
struct ChordTone {
    pitch: Pitch,
    dur: u32,
}

/// One melody event: a note or a rest, plus everything attached to it.
/// Chord notes do not advance time and are stored on the event they sound with.
struct Event {
    dur: u32,
    tone: Option<Pitch>,
    chords: Vec<ChordTone>,
    decos: Vec<String>,
    annos: Vec<String>,
    slur_starts: Vec<u8>,
    slur_stops: Vec<u8>,
    tie_start: bool,
    tie_stop: bool,
}

struct MeasureData {
    /// Zero-based measure index within the part.
    idx: usize,
    number: String,
    events: Vec<Event>,
    /// The right barline is a double bar (`||`).
    bar_double: bool,
    key_change: Option<i8>,
    time_change: Option<(u32, u32)>,
    multirest: Option<u32>,
    /// Part of a collapsed multi-rest block: not emitted.
    skip: bool,
    /// Length of one beat (per the time signature) in divisions.
    beat_div: u32,
    beats: u32,
}

/// A key section: the span between key signature changes, with the jianpu
/// tonic (1 = ...) chosen for the section.
struct Section {
    start: usize, // measure index
    fifths: i8,
    tonic_letter: usize,
    tonic_alter: i16,
    tonic_octave: u8,
}

fn step_index(s: &Step) -> usize {
    match s {
        Step::C => 0,
        Step::D => 1,
        Step::E => 2,
        Step::F => 3,
        Step::G => 4,
        Step::A => 5,
        Step::B => 6,
    }
}

/// Tonic (letter index, alteration) of the major key with `fifths` sharps
/// (positive) or flats (negative).
fn key_tonic(fifths: i8) -> (usize, i16) {
    match fifths {
        0 => (0, 0),   // C
        1 => (4, 0),   // G
        2 => (1, 0),   // D
        3 => (5, 0),   // A
        4 => (2, 0),   // E
        5 => (6, 0),   // B
        6 => (3, 1),   // F#
        7 => (0, 1),   // C#
        -1 => (3, 0),  // F
        -2 => (6, -1), // Bb
        -3 => (2, -1), // Eb
        -4 => (5, -1), // Ab
        -5 => (1, -1), // Db
        -6 => (4, -1), // Gb
        -7 => (0, -1), // Cb
        _ => (0, 0),
    }
}

/// Alteration a step receives from the key signature alone.
fn key_default_alter(fifths: i8, step: usize) -> i16 {
    if fifths > 0 && SHARP_ORDER[..fifths as usize].contains(&step) {
        1
    } else if fifths < 0 && FLAT_ORDER[..(-fifths) as usize].contains(&step) {
        -1
    } else {
        0
    }
}

/// Diatonic distance in octaves from the tonic: 0 = within the unmarked
/// octave, +1 = one apostrophe up, -1 = one comma down.
fn octave_marks(pitch: &Pitch, sec: &Section) -> i32 {
    let d = (pitch.octave as i32 * 7 + pitch.step as i32)
        - (sec.tonic_octave as i32 * 7 + sec.tonic_letter as i32);
    if d >= 0 {
        d / 7
    } else {
        -((-d + 6) / 7)
    }
}

/// Map a duration in divisions to (subtraction-line/dot suffix, dash count).
/// Durations are measured against one beat (`beat_div` divisions).
/// Returns None for durations Tomato cannot express.
fn duration_parts(dur: u32, beat_div: u32) -> Option<(&'static str, usize)> {
    match dur {
        d if d == beat_div => Some(("", 0)),            // 1 beat
        d if d * 2 == beat_div => Some(("/", 0)),      // 1/2 beat
        d if d * 4 == beat_div => Some(("//", 0)),     // 1/4 beat
        d if d * 4 == beat_div * 3 => Some(("/.", 0)), // 3/4 beat (dot after slash)
        d if d * 2 == beat_div * 3 => Some((".", 0)),  // 1 1/2 beats
        d if d == beat_div * 2 => Some(("", 1)),
        d if d == beat_div * 3 => Some(("", 2)),
        d if d == beat_div * 4 => Some(("", 3)),
        _ => None,
    }
}

/// Rest tokens: whole beats expand to one `0` per beat (the example script
/// writes full-measure rests as `0 0 0 0`), fractions map like notes.
fn rest_tokens(dur: u32, beat_div: u32) -> Option<Vec<String>> {
    if dur.is_multiple_of(beat_div) {
        Some(vec!["0".to_string(); (dur / beat_div) as usize])
    } else {
        duration_parts(dur, beat_div).map(|(suffix, _)| vec![format!("0{suffix}")])
    }
}

/// Note tokens: degree digit + accidental + octave marks + duration suffix,
/// followed by the augmentation-line tokens (dashes).
fn note_tokens(
    pitch: &Pitch,
    dur: u32,
    beat_div: u32,
    sec: &Section,
    measure_number: &str,
) -> Option<Vec<String>> {
    let degree = (pitch.step + 7 - sec.tonic_letter) % 7 + 1;
    let diff = pitch.alter - key_default_alter(sec.fifths, pitch.step);
    let accidental = match diff {
        0 => "",
        1 => "#",
        -1 if pitch.alter == 0 => "=",
        -1 => "$",
        _ => {
            eprintln!("[warn] measure {measure_number}: alteration of {diff} semitones is not supported, skipped");
            ""
        }
    };
    let k = octave_marks(pitch, sec);
    let octave = if k > 0 {
        "'".repeat(k as usize)
    } else {
        ",".repeat((-k) as usize)
    };
    let (suffix, dashes) = match duration_parts(dur, beat_div) {
        Some(parts) => parts,
        None => {
            eprintln!("[warn] measure {measure_number}: duration of {dur} divisions is not representable in jianpu, written as a quarter note");
            ("", 0)
        }
    };
    let mut tokens = vec![format!("{degree}{accidental}{octave}{suffix}")];
    tokens.extend(std::iter::repeat_n("-".to_string(), dashes));
    Some(tokens)
}

fn decode_dynamics(d: &DynamicsType, measure_number: &str) -> Option<String> {
    let code = match d {
        DynamicsType::P(_) => "&p",
        DynamicsType::Pp(_) => "&pp",
        DynamicsType::Ppp(_) => "&ppp",
        DynamicsType::Mp(_) => "&mp",
        DynamicsType::Mf(_) => "&mf",
        DynamicsType::F(_) => "&f",
        DynamicsType::Ff(_) => "&ff",
        DynamicsType::Fff(_) => "&fff",
        _ => {
            eprintln!("[warn] measure {measure_number}: unsupported dynamic, skipped");
            return None;
        }
    };
    Some(code.to_string())
}

/// Decode a `<direction>` into decoration codes to attach to the next note.
fn decode_direction(dir: &musicxml::elements::Direction, measure_number: &str) -> Vec<String> {
    let mut out = Vec::new();
    for dt in &dir.content.direction_type {
        match &dt.content {
            DirectionTypeContents::Dynamics(list) => {
                for d in list {
                    for kind in &d.content {
                        if let Some(code) = decode_dynamics(kind, measure_number) {
                            out.push(code);
                        }
                    }
                }
            }
            DirectionTypeContents::Wedge(w) => match w.attributes.r#type {
                WedgeType::Crescendo => out.push("&<".to_string()),
                WedgeType::Diminuendo => out.push("&>".to_string()),
                WedgeType::Stop => out.push("&!".to_string()),
                WedgeType::Continue => {}
            },
            // Words (tempo/text directions) have no jianpu equivalent here.
            _ => {}
        }
    }
    out
}

/// Walk the first part of the score and flatten it into per-measure events.
fn collect_measures(score: &ScorePartwise) -> (Vec<MeasureData>, String, (u32, u32)) {
    let part = &score.content.part[0];

    let title = score
        .content
        .movement_title
        .as_ref()
        .map(|t| t.content.clone())
        .or_else(|| {
            score
                .content
                .work
                .as_ref()
                .and_then(|w| w.content.work_title.as_ref().map(|t| t.content.clone()))
        })
        .or_else(|| part_list_name(score))
        .unwrap_or_else(|| "Untitled".to_string());

    let mut measures = Vec::new();
    let mut divisions = 1u32;
    let mut fifths: Option<i8> = None;
    let mut time: Option<(u32, u32)> = None;
    let mut initial_time = (4u32, 4u32);

    for (idx, part_el) in part.content.iter().enumerate() {
        let PartElement::Measure(m) = part_el else {
            continue;
        };
        let number = m.attributes.number.0.clone();

        let mut data = MeasureData {
            idx,
            number: number.clone(),
            events: Vec::new(),
            bar_double: false,
            key_change: None,
            time_change: None,
            multirest: None,
            skip: false,
            beat_div: 0,
            beats: 0,
        };

        let mut pending: Vec<String> = Vec::new();

        // First pass: apply attributes so the whole measure uses the
        // divisions/key/time they set.
        for el in &m.content {
            if let MeasureElement::Attributes(a) = el {
                if let Some(ref d) = a.content.divisions {
                    divisions = d.content.0;
                }
                for k in &a.content.key {
                    if let KeyContents::Explicit(e) = &k.content {
                        let f = e.fifths.content.0;
                        // Only major keys occur in the input; the mode
                        // element is absent and keys are treated as major.
                        if fifths != Some(f) {
                            // The first key also becomes a section (at
                            // measure 0) so every measure has a section.
                            data.key_change = Some(f);
                            fifths = Some(f);
                        }
                    }
                }
                for t in &a.content.time {
                    // TimeBeatContents is one (beats, beat-type) pair;
                    // multiple pairs would mean a composite time signature.
                    if let Some(b) = t.content.beats.first()
                        && let (Ok(beats_new), Ok(bt_new)) = (
                            b.beats.content.trim().parse::<u32>(),
                            b.beat_type.content.trim().parse::<u32>(),
                        )
                    {
                        let new_time = (beats_new, bt_new);
                        if time != Some(new_time) {
                            if time.is_none() {
                                initial_time = new_time;
                            } else {
                                data.time_change = Some(new_time);
                            }
                            time = Some(new_time);
                        }
                    }
                }
                for ms in &a.content.measure_style {
                    if let MeasureStyleContents::MultipleRest(mr) = &ms.content {
                        data.multirest = Some(mr.content.0);
                    }
                }
            }
        }

        // Beat length in divisions: a beat is 4/beat_type quarter notes.
        // Fall back to one division if the time signature is unusable.
        let (beats, beat_type) = time.unwrap_or((4, 4));
        data.beats = beats;
        data.beat_div = divisions
            .checked_mul(4)
            .and_then(|per_quarter| per_quarter.checked_div(beat_type))
            .unwrap_or(divisions);

        // Second pass: notes, directions, barlines.
        for el in &m.content {
            match el {
                MeasureElement::Attributes(_) => {}
                MeasureElement::Direction(d) => {
                    pending.extend(decode_direction(d, &number));
                }
                MeasureElement::Note(note) => {
                    let NoteType::Normal(n) = &note.content.info else {
                        eprintln!("[warn] measure {number}: grace/cue note is not supported, skipped");
                        continue;
                    };
                    let dur = n.duration.content.0;
                    let tone = match &n.audible {
                        AudibleType::Pitch(p) => Some(Pitch {
                            step: step_index(&p.content.step.content),
                            alter: p
                                .content
                                .alter
                                .as_ref()
                                .map(|a| a.content.0)
                                .unwrap_or(0),
                            octave: p.content.octave.content.0,
                        }),
                        AudibleType::Rest(_) => None,
                        AudibleType::Unpitched(_) => {
                            eprintln!("[warn] measure {number}: unpitched note is not supported, skipped");
                            None
                        }
                    };

                    let mut decos: Vec<String> = Vec::new();
                    let mut slur_starts = Vec::new();
                    let mut slur_stops = Vec::new();
                    let mut tie_start = false;
                    let mut tie_stop = false;

                    for t in &n.tie {
                        match t.attributes.r#type {
                            StartStop::Start => tie_start = true,
                            StartStop::Stop => tie_stop = true,
                        }
                    }
                    for nots in &note.content.notations {
                        for nc in &nots.content.notations {
                            match nc {
                                NotationContentTypes::Slur(s) => {
                                    let num = s.attributes.number.as_ref().map(|n| n.0).unwrap_or(1);
                                    match s.attributes.r#type {
                                        StartStopContinue::Start => slur_starts.push(num),
                                        StartStopContinue::Stop => slur_stops.push(num),
                                        StartStopContinue::Continue => {}
                                    }
                                }
                                // Tied (the notation form of a tie) is ignored;
                                // the sounding Tie element is used instead.
                                NotationContentTypes::Articulations(arts) => {
                                    for art in &arts.content {
                                        let code = match art {
                                            ArticulationsType::Accent(_) => "&zy",
                                            ArticulationsType::Tenuto(_) => "&bc",
                                            ArticulationsType::Staccato(_) => "&dy",
                                            _ => {
                                                eprintln!("[warn] measure {number}: unsupported articulation, skipped");
                                                continue;
                                            }
                                        };
                                        decos.push(code.to_string());
                                    }
                                }
                                NotationContentTypes::Ornaments(o) => {
                                    for orn in &o.content.ornaments {
                                        let code = match orn {
                                            OrnamentType::TrillMark(_) => "&cy",
                                            OrnamentType::InvertedMordent(_) => "&xby",
                                            OrnamentType::Mordent(_) => "&sby",
                                            _ => {
                                                eprintln!("[warn] measure {number}: unsupported ornament, skipped");
                                                continue;
                                            }
                                        };
                                        decos.push(code.to_string());
                                    }
                                }
                                NotationContentTypes::Fermata(_) => decos.push("&yc".to_string()),
                                NotationContentTypes::Tuplet(_) => {
                                    eprintln!("[warn] measure {number}: tuplet is not supported, skipped");
                                }
                                _ => {}
                            }
                        }
                    }

                    if n.chord.is_some() {
                        // Chord notes do not advance time; attach to the event
                        // they sound with. Their slurs/ties are dropped: the
                        // main note already carries the phrasing.
                        match (&mut data.events.last_mut(), tone) {
                            (Some(last), Some(pitch)) => {
                                last.chords.push(ChordTone { pitch, dur });
                            }
                            _ => eprintln!("[warn] measure {number}: chord on a rest or empty measure, skipped"),
                        }
                    } else {
                        let mut all_decos = std::mem::take(&mut pending);
                        all_decos.extend(decos);
                        data.events.push(Event {
                            dur,
                            tone,
                            chords: Vec::new(),
                            decos: all_decos,
                            annos: Vec::new(),
                            slur_starts,
                            slur_stops,
                            tie_start,
                            tie_stop,
                        });
                    }
                }
                MeasureElement::Barline(b) => {
                    // location defaults to "right"; only right barlines matter.
                    if !matches!(b.attributes.location, Some(RightLeftMiddle::Left))
                        && let Some(style) = &b.content.bar_style
                    {
                        data.bar_double = matches!(
                            style.content,
                            BarStyle::LightLight | BarStyle::LightHeavy | BarStyle::HeavyHeavy
                        );
                        // The v1 spec table has no heavy barline symbol, so
                        // final (light-heavy) barlines are written as `||`.
                        if matches!(
                            style.content,
                            BarStyle::Dashed
                                | BarStyle::Dotted
                                | BarStyle::Heavy
                                | BarStyle::HeavyLight
                                | BarStyle::Short
                                | BarStyle::Tick
                        ) {
                            eprintln!("[warn] measure {number}: bar style {style:?} has no jianpu equivalent, written as a plain barline");
                        }
                    }
                }
                MeasureElement::Backup(_) | MeasureElement::Forward(_) => {
                    eprintln!("[warn] measure {number}: backup/forward (multiple voices) is not supported, skipped");
                }
                _ => {}
            }
        }

        // Directions left at the end of a measure (e.g. a wedge stop) attach
        // to the last note of the measure.
        if !pending.is_empty()
            && let Some(last) = data.events.last_mut()
        {
            last.decos.extend(pending);
        }
        measures.push(data);
    }

    (measures, title, initial_time)
}

fn part_list_name(score: &ScorePartwise) -> Option<String> {
    score
        .content
        .part_list
        .content
        .content
        .iter()
        .find_map(|el| {
            let PartListElement::ScorePart(sp) = el else {
                return None;
            };
            Some(sp.content.part_name.content.clone())
        })
}

/// Build the key sections and choose the tonic octave for each: the octave
/// whose unmarked range minimizes the total number of octave marks over the
/// section's notes, so the melody sits "around" 1 as much as possible.
fn build_sections(measures: &[MeasureData]) -> Vec<Section> {
    let mut starts: Vec<(usize, i8)> = Vec::new();
    for m in measures {
        if let Some(f) = m.key_change
            && starts.last().map(|(_, pf)| *pf) != Some(f)
        {
            starts.push((m.idx, f));
        }
    }

    let mut sections = Vec::new();
    for (si, &(start, fifths)) in starts.iter().enumerate() {
        let end = starts
            .get(si + 1)
            .map(|&(s, _)| s)
            .unwrap_or_else(|| measures.len());
        let (tonic_letter, tonic_alter) = key_tonic(fifths);
        let notes: Vec<Pitch> = measures[start..end]
            .iter()
            .flat_map(|m| &m.events)
            .filter_map(|ev| ev.tone)
            .collect();
        let mut best = (u64::MAX, 4u8);
        for toct in 1..=8u8 {
            let probe = Section {
                start,
                fifths,
                tonic_letter,
                tonic_alter,
                tonic_octave: toct,
            };
            let marks: u64 = notes
                .iter()
                .map(|p| octave_marks(p, &probe).unsigned_abs() as u64)
                .sum();
            if marks < best.0 {
                best = (marks, toct);
            }
        }
        sections.push(Section {
            start,
            fifths,
            tonic_letter,
            tonic_alter,
            tonic_octave: best.1,
        });
    }
    sections
}

fn section_for(sections: &[Section], measure_idx: usize) -> &Section {
    sections
        .iter()
        .rev()
        .find(|s| s.start <= measure_idx)
        .unwrap_or_else(|| &sections[0])
}

/// State for slur/tie parentheses, carried across measures and lines.
#[derive(Default)]
struct ParenState {
    slurs_open: HashSet<u8>,
    tie_open: bool,
}

impl ParenState {
    /// Consume an event's slur/tie flags, returning (opens, closes) and
    /// updating the state. Tie chains (stop+start on one note) keep the
    /// paren open, so a chain renders as one slur region.
    fn consume(&mut self, ev: &Event) -> (usize, usize) {
        let mut opens = 0;
        let mut closes = 0;
        for n in &ev.slur_stops {
            if self.slurs_open.remove(n) {
                closes += 1;
            }
        }
        for n in &ev.slur_starts {
            if self.slurs_open.insert(*n) {
                opens += 1;
            }
        }
        match (ev.tie_stop, ev.tie_start) {
            (true, true) => {}
            (true, false) => {
                if self.tie_open {
                    self.tie_open = false;
                    closes += 1;
                }
            }
            (false, true) => {
                if !self.tie_open {
                    self.tie_open = true;
                    opens += 1;
                }
            }
            (false, false) => {}
        }
        (opens, closes)
    }
}

/// Does this line hold a long enough chord run to warrant two staves?
fn line_needs_two_staves<'a>(events: impl IntoIterator<Item = &'a Event>) -> bool {
    let mut run = 0;
    let mut best = 0;
    let mut total = 0;
    for ev in events {
        if !ev.chords.is_empty() {
            run += 1;
            total += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best >= CHORD_RUN_FOR_TWO_STAVES || total >= CHORD_COUNT_FOR_TWO_STAVES
}

fn annotate(mut token: String, ev: &Event) -> String {
    for d in &ev.decos {
        token.push_str(d);
    }
    for a in &ev.annos {
        token.push('"');
        token.push_str(a);
        token.push('"');
    }
    token
}

/// Render one chunk of measures into one or two output lines.
/// Single mode emits `Q: ...`; two-staff mode emits `Q1: ...` and `Q2: ...`.
fn render_chunk(
    chunk: &[&MeasureData],
    sections: &[Section],
    paren: &mut ParenState,
) -> Vec<String> {
    let two_staves = line_needs_two_staves(chunk.iter().flat_map(|m| &m.events));

    if !two_staves {
        vec![render_staff(chunk, sections, Staff::Main, paren)]
    } else {
        vec![
            render_staff(chunk, sections, Staff::Upper, paren),
            render_staff(chunk, sections, Staff::Lower, paren),
        ]
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Staff {
    /// `Q:` single line, or `Q1:` in two-stave mode (the melody).
    Main,
    Upper,
    /// `Q2:` in two-stave mode: chord tones, rests where the melody has none.
    Lower,
}

fn render_staff(
    chunk: &[&MeasureData],
    sections: &[Section],
    staff: Staff,
    paren: &mut ParenState,
) -> String {
    let prefix = match staff {
        Staff::Main => "Q:",
        Staff::Upper => "Q1:",
        Staff::Lower => "Q2:",
    };
    // Parentheses belong to the melody staff only; the lower staff carries
    // bare chord tones.
    let with_parens = !matches!(staff, Staff::Lower);

    let mut parts: Vec<String> = vec![prefix.to_string()];

    // A time signature change at the first measure goes on the leading bar.
    let lead = match chunk[0].time_change {
        Some((b, t)) => format!("|\"p:{b}/{t}\""),
        None => "|".to_string(),
    };
    parts.push(lead);

    for (i, m) in chunk.iter().enumerate() {
        let sec = section_for(sections, m.idx);
        for ev in &m.events {
            // Isolated chords: overlay them with {dsb ...} before the note.
            if matches!(staff, Staff::Main) && !ev.chords.is_empty() {
                let mut overlay: Vec<String> = Vec::new();
                for ct in &ev.chords {
                    if let Some(toks) = note_tokens(&ct.pitch, ct.dur, m.beat_div, sec, &m.number) {
                        overlay.extend(toks);
                    }
                }
                parts.push(format!("{{dsb {} |}}", overlay.join(" ")));
            }

            let (opens, closes) = if with_parens {
                paren.consume(ev)
            } else {
                (0, 0)
            };
            let parens_open = "(".repeat(opens);
            let parens_close = ")".repeat(closes);

            let mut toks = match (staff, &ev.tone) {
                (Staff::Lower, _) => {
                    // Lower staff: chord tones here, rests where the melody
                    // has none.
                    if ev.chords.is_empty() {
                        rest_tokens(ev.dur, m.beat_div)
                    } else {
                        let mut all = Vec::new();
                        for ct in &ev.chords {
                            if let Some(t) = note_tokens(&ct.pitch, ct.dur, m.beat_div, sec, &m.number)
                            {
                                all.extend(t);
                            }
                        }
                        if all.is_empty() {
                            None
                        } else {
                            Some(all)
                        }
                    }
                }
                (_, Some(p)) => note_tokens(p, ev.dur, m.beat_div, sec, &m.number),
                (_, None) => rest_tokens(ev.dur, m.beat_div),
            };

            if let Some(toks) = &mut toks {
                if !toks.is_empty() {
                    let first = annotate(format!("{parens_open}{}", toks[0]), ev);
                    toks[0] = format!("{first}{parens_close}");
                }
                parts.extend(toks.iter().cloned());
            }
        }
        // Barline closing the measure; a time change in the next measure of
        // the same line is noted on it.
        let mut bar = if m.bar_double { "||" } else { "|" }.to_string();
        if let Some(next) = chunk.get(i + 1)
            && let Some((b, t)) = next.time_change
        {
            bar.push_str(&format!("\"p:{b}/{t}\""));
        }
        parts.push(bar);
    }
    parts.join(" ")
}

/// Verify that a measure's durations add up to its time signature.
fn check_measure(m: &MeasureData) {
    let expected = m.beats * m.beat_div;
    let actual: u32 = m.events.iter().map(|ev| ev.dur).sum();
    if actual != expected {
        eprintln!(
            "[warn] measure {}: durations add up to {actual} divisions, expected {expected}",
            m.number
        );
    }
}

/// Collapse multi-rest blocks that span more than `COLLAPSE_MULTIREST_ABOVE`
/// measures: mark the continuation measures to skip and carry the block's
/// closing barline onto the head measure, which keeps its "N小節" annotation.
fn collapse_multirests(measures: &mut [MeasureData]) {
    for i in 0..measures.len() {
        let Some(n) = measures[i].multirest else {
            continue;
        };
        if n <= COLLAPSE_MULTIREST_ABOVE {
            continue;
        }
        // Walk the block's continuation measures; stop at anything that is
        // not a plain whole-measure rest or that carries a change.
        let mut last = i;
        while last + 1 < measures.len()
            && measures[last + 1].idx - measures[i].idx < n as usize
            && measures[last + 1].key_change.is_none()
            && measures[last + 1].time_change.is_none()
            && measures[last + 1].multirest.is_none()
            && !measures[last + 1].skip
            && measures[last + 1].events.len() == 1
            && measures[last + 1].events[0].tone.is_none()
            && measures[last + 1].events[0].chords.is_empty()
        {
            last += 1;
        }
        if last == i {
            continue;
        }
        let end_bar = measures[last].bar_double;
        measures[i].bar_double = end_bar;
        for m in &mut measures[i + 1..=last] {
            m.skip = true;
        }
    }
}

fn main() {
    let input = env::args()
        .nth(1)
        .unwrap_or_else(|| "bangdi_cleaned.musicxml".to_string());

    let score = match read_score_partwise(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {input}: {e}");
            std::process::exit(1);
        }
    };

    let (mut measures, title, time) = collect_measures(&score);
    for m in &measures {
        check_measure(m);
    }
    collapse_multirests(&mut measures);
    let sections = build_sections(&measures);

    // Header. The tonic letter carries # or $ for altered keys; the tempo
    // field (J:) is omitted because the input has no tempo mark, and the
    // author field (Z:) is omitted because the input names none.
    let fifths = sections[0].fifths;
    let (letter, tonic_alter) = key_tonic(fifths);
    let key_symbol = match tonic_alter {
        1 => "#",
        -1 => "$",
        _ => "",
    };
    let mut out = String::new();
    out.push_str("V: 1.0\n");
    out.push_str(&format!("B: {title}\n"));
    out.push_str(&format!("D: {}{key_symbol}\n", STEP_LETTERS[letter]));
    out.push_str(&format!("P: {}/{}\n", time.0, time.1));

    // Attach annotations that live on a measure's first note: key changes
    // ("1=X", matching the example script's style) and multiple-rest counts.
    // The initial key is already in the D: header, so measure 0 is skipped.
    for m in measures.iter_mut() {
        if m.idx > 0
            && let (Some(first), Some(sec)) = (
                m.events.first_mut(),
                sections.iter().find(|s| s.start == m.idx),
            )
        {
            let sym = match sec.tonic_alter {
                1 => "#",
                -1 => "b",
                _ => "",
            };
            first
                .annos
                .push(format!("1={}{}", STEP_LETTERS[sec.tonic_letter], sym));
        }
        if let (Some(first), Some(n)) = (m.events.first_mut(), m.multirest)
            && n > COLLAPSE_MULTIREST_ABOVE
        {
            first.annos.push(format!("{n}小節"));
        }
    }

    // Measures hidden inside collapsed multi-rest blocks are not emitted.
    let output: Vec<&MeasureData> = measures.iter().filter(|m| !m.skip).collect();
    let chunks: Vec<&[&MeasureData]> = output.chunks(MEASURES_PER_LINE).collect();
    let mut paren = ParenState::default();
    let mut chunk_lines: Vec<Vec<String>> = Vec::new();
    for chunk in &chunks {
        chunk_lines.push(render_chunk(chunk, &sections, &mut paren));
    }
    // Tomato only paginates on the manual [fenye] marker; the example script
    // fits about 12 lines on an A4 page. Page breaks go between chunks: a
    // Q1:/Q2: staff pair is one system and must not be split.
    const LINES_PER_PAGE: usize = 12;
    let mut paginated: Vec<String> = Vec::new();
    let mut page_fill = 0usize;
    for chunk_lines in &chunk_lines {
        if page_fill > 0 && page_fill + chunk_lines.len() > LINES_PER_PAGE {
            paginated.push("[fenye]".to_string());
            page_fill = 0;
        }
        paginated.extend(chunk_lines.iter().cloned());
        page_fill += chunk_lines.len();
    }
    out.push_str(&paginated.join("\n"));
    out.push('\n');

    let out_path = Path::new(&input).with_extension("jps");
    if let Err(e) = fs::write(&out_path, &out) {
        eprintln!("error: cannot write {}: {e}", out_path.display());
        std::process::exit(1);
    }
    println!("wrote {}", out_path.display());
}
