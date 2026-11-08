// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Knobs: what a run sets without a rebuild.
//!
//! A scenario declares its knobs, each with a class, bounds and a default that
//! equals the number its code held before. A run sets them in `LANCE_STRESS_SET`,
//! a comma-separated list of `<knob>=<value>`, or in a file that
//! `LANCE_STRESS_SET_FILE` names, one pair per line and `#` for a comment. A key
//! without a prefix sets the knob of that name of every scenario that has one,
//! `<scenario>.<knob>` that of one scenario. A few older variables set a knob too,
//! as short forms, read as they always were: an integer, in seconds for a length.
//!
//! What a variable sets wins over the file, and within one source a key with a
//! prefix wins over one without. A knob set both by a short form and in
//! `LANCE_STRESS_SET`, the same key twice in one source, a key no scenario has, and
//! a value of the wrong kind or out of bounds all fail every test of the run, in
//! the parent, before a child starts: a typo must not pass for a run that checked
//! nothing.
//!
//! The parent resolves the knobs of a scenario once and gives each child the whole
//! of them as one record in `LANCE_STRESS_SET`, the only source of knobs a child
//! reads; the steps of its plan come resolved in `LANCE_STRESS_STEPS`. The replay of
//! a seed pins the same record and removes from its shell the sources of what the
//! record leaves out, so it does not depend on the shell it runs in; a stall or a
//! deadline that no source set stays out of it, and the replay derives it again. A knob of the class `plan` that is off its default
//! goes into the plan, and so into its fingerprint; `summary.txt` names every knob
//! off its default.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::time::Duration;

use crate::harness::ENV_STEPS;
use crate::scenario::soak::ENV_SOAK_SECS;
use crate::scenario::{REGISTRY, Scenario, SelfTest};

pub const ENV_SET: &str = "LANCE_STRESS_SET";
pub const ENV_SET_FILE: &str = "LANCE_STRESS_SET_FILE";
pub const ENV_STALL_SECS: &str = "LANCE_STRESS_STALL_SECS";
pub const ENV_DEADLINE_SECS: &str = "LANCE_STRESS_DEADLINE_SECS";
/// The sources of what the record of a replay leaves out, a stall or a deadline
/// that no source set: its line removes them from the shell, where they would set
/// what the replay derives again. A knob the record pins conflicts with a short
/// form instead.
pub const UNPINNED: [&str; 3] = [ENV_SET_FILE, ENV_STALL_SECS, ENV_DEADLINE_SECS];

/// The older variables that set a knob, the scenario they set it for, if one, and
/// the unit their integer is in.
const SHORT_FORMS: [(&str, Option<&str>, &str, &str); 7] = [
    ("LANCE_STRESS_SEEDS", None, "seeds", ""),
    ("LANCE_STRESS_SEED_BASE", None, "seed_base", ""),
    ("LANCE_STRESS_REPEAT", None, "repeat", ""),
    (ENV_STEPS, None, "steps", ""),
    (ENV_STALL_SECS, None, "stall", "s"),
    (ENV_DEADLINE_SECS, None, "deadline", "s"),
    (ENV_SOAK_SECS, Some("soak"), "secs", "s"),
];

const MILLIS: [(&str, u64); 5] = [
    ("d", 86_400_000),
    ("h", 3_600_000),
    ("m", 60_000),
    ("s", 1000),
    ("ms", 1),
];

const BYTES: [(&str, u64); 4] = [
    ("GiB", 1 << 30),
    ("MiB", 1 << 20),
    ("KiB", 1 << 10),
    ("B", 1),
];

/// The variables a parent reads its knobs and modes from: those of the process,
/// or what a test hands it, so that no test sets a variable of the process.
#[derive(Clone, Debug, Default)]
pub struct Env(BTreeMap<String, String>);

impl Env {
    pub fn of_process() -> Self {
        Self::of(std::env::vars_os())
    }

    fn of(os_vars: impl IntoIterator<Item = (OsString, OsString)>) -> Self {
        let mut vars = BTreeMap::new();
        for (name, value) in os_vars {
            let Ok(name) = name.into_string() else {
                continue;
            };
            match value.into_string() {
                Ok(value) => {
                    vars.insert(name, value);
                }
                // The harness reads its own, and a replay repeats those of Lance.
                Err(value) => assert!(
                    !name.starts_with("LANCE_"),
                    "{name} is not UTF-8: {value:?}"
                ),
            }
        }
        Self(vars)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    /// The variables of Lance that the harness does not set itself, by name.
    pub fn lance_tuning(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .filter(|(name, _)| name.starts_with("LANCE_") && !name.starts_with("LANCE_STRESS_"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
    }
}

impl<'a> FromIterator<(&'a str, String)> for Env {
    fn from_iter<I: IntoIterator<Item = (&'a str, String)>>(vars: I) -> Self {
        Self(
            vars.into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
        )
    }
}

impl<const N: usize> From<[(&str, &str); N]> for Env {
    fn from(vars: [(&str, &str); N]) -> Self {
        Self(
            vars.iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        )
    }
}

/// What a knob is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// How many seeds a run takes, and from which. A replay runs one seed, so its
    /// command leaves these out.
    Shape,
    /// What a seed runs. One off its default goes into the plan.
    Plan,
    /// A threshold of a verdict, or of the parent's patience.
    Judge,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Self::Shape => "shape",
            Self::Plan => "plan",
            Self::Judge => "judge",
        }
    }
}

/// The values a knob takes, within its bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Count {
        least: u64,
        most: u64,
    },
    /// `a..=b`, both ends within the bounds; `a` alone is `a..=a`.
    Range {
        least: u64,
        most: u64,
    },
    Duration {
        least: Duration,
        most: Duration,
    },
    Bytes {
        least: u64,
        most: u64,
    },
    /// A probability or a multiplier, as a decimal.
    Share {
        least: f64,
        most: f64,
    },
}

/// A knob's value. It prints as a key's value is written: a length or a size in
/// the largest unit that holds it exactly, a share as the shortest decimal that
/// reads back as the same number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Count(u64),
    Range(u64, u64),
    Duration(Duration),
    Bytes(u64),
    Share(f64),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn scaled(f: &mut fmt::Formatter<'_>, amount: u64, units: &[(&str, u64)]) -> fmt::Result {
            let (unit, size) = units
                .iter()
                .find(|(_, size)| amount.is_multiple_of(*size) && (amount > 0 || *size == 1))
                .expect("the last unit is 1");
            write!(f, "{}{unit}", amount / size)
        }
        match *self {
            Self::Count(count) => write!(f, "{count}"),
            Self::Range(start, end) => write!(f, "{start}..={end}"),
            Self::Duration(length) => scaled(f, length.as_millis() as u64, &MILLIS),
            Self::Bytes(bytes) => scaled(f, bytes, &BYTES),
            Self::Share(share) => write!(f, "{share}"),
        }
    }
}

impl Kind {
    fn bounds(&self) -> String {
        let (least, most) = match *self {
            Self::Count { least, most } | Self::Range { least, most } => {
                (Value::Count(least), Value::Count(most))
            }
            Self::Duration { least, most } => (Value::Duration(least), Value::Duration(most)),
            Self::Bytes { least, most } => (Value::Bytes(least), Value::Bytes(most)),
            Self::Share { least, most } => (Value::Share(least), Value::Share(most)),
        };
        format!("{least}..={most}")
    }

    /// Reads `text` as a value of this kind, within the bounds.
    fn parse(&self, text: &str) -> Result<Value, String> {
        let value = self
            .read(text)
            .map_err(|reason| format!("{reason}; it takes {}", self.bounds()))?;
        let within = match (*self, value) {
            (Self::Count { least, most }, Value::Count(count)) => (least..=most).contains(&count),
            (Self::Range { least, most }, Value::Range(start, end)) => {
                (least..=most).contains(&start) && (least..=most).contains(&end)
            }
            (Self::Duration { least, most }, Value::Duration(length)) => {
                (least..=most).contains(&length)
            }
            (Self::Bytes { least, most }, Value::Bytes(bytes)) => (least..=most).contains(&bytes),
            (Self::Share { least, most }, Value::Share(share)) => (least..=most).contains(&share),
            _ => unreachable!("a kind parses its own values"),
        };
        match within {
            true => Ok(value),
            false => Err(format!("is out of its bounds {}", self.bounds())),
        }
    }

    /// Reads `text` as a value of this kind, whatever the bounds.
    fn read(&self, text: &str) -> Result<Value, String> {
        Ok(match *self {
            Self::Count { .. } => Value::Count(parse_count(text)?),
            Self::Range { .. } => {
                if text.contains("..") && !text.contains("..=") {
                    return Err("is not a range: write `a..=b`, both ends included".to_string());
                }
                let (start, end) = text.split_once("..=").unwrap_or((text, text));
                let (start, end) = (parse_count(start)?, parse_count(end)?);
                if start > end {
                    return Err("is an empty range".to_string());
                }
                Value::Range(start, end)
            }
            Self::Duration { .. } => Value::Duration(Duration::from_millis(parse_scaled(
                text, &MILLIS, "a length",
            )?)),
            Self::Bytes { .. } => Value::Bytes(parse_scaled(text, &BYTES, "a size")?),
            Self::Share { .. } => Value::Share(parse_share(text)?),
        })
    }
}

fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn parse_count(text: &str) -> Result<u64, String> {
    if !is_digits(text) {
        return Err("is not an unsigned integer".to_string());
    }
    text.parse().map_err(|_| "is too large".to_string())
}

/// A decimal with a unit, exactly a whole number of the smallest unit.
fn parse_scaled(text: &str, units: &[(&str, u64)], what: &str) -> Result<u64, String> {
    let names: Vec<&str> = units.iter().map(|(unit, _)| *unit).collect();
    // The longest unit first: `ms` ends in `s`, `KiB` in `B`.
    let mut by_length = units.to_vec();
    by_length.sort_by_key(|(unit, _)| std::cmp::Reverse(unit.len()));
    let (number, size) = by_length
        .iter()
        .find_map(|(unit, size)| text.strip_suffix(unit).map(|number| (number, *size)))
        .ok_or_else(|| format!("is {what} without a unit, one of {}", names.join(", ")))?;
    let (whole, fraction) = number.split_once('.').unwrap_or((number, "0"));
    if !is_digits(whole) || !is_digits(fraction) {
        return Err(format!("is not {what}"));
    }
    let too_large = || "is too large".to_string();
    // Zeros at the end of a fraction add nothing to it.
    let fraction = fraction.trim_end_matches('0');
    let part = match fraction.is_empty() {
        true => 0,
        false => {
            let long = || "has too many digits after the point".to_string();
            let digits: u128 = fraction.parse().map_err(|_| long())?;
            let scale = 10u128.checked_pow(fraction.len() as u32).ok_or_else(long)?;
            let scaled = digits.checked_mul(u128::from(size)).ok_or_else(long)?;
            if scaled % scale != 0 {
                return Err(format!(
                    "is finer than its smallest unit, {}",
                    names[names.len() - 1]
                ));
            }
            u64::try_from(scaled / scale).map_err(|_| too_large())?
        }
    };
    whole
        .parse::<u64>()
        .ok()
        .and_then(|whole| whole.checked_mul(size))
        .and_then(|amount| amount.checked_add(part))
        .ok_or_else(too_large)
}

/// A decimal, read as the literal of the same digits would be, so a default
/// written `0.7` and a value `0.7` are the same number.
fn parse_share(text: &str) -> Result<f64, String> {
    let (whole, fraction) = text.split_once('.').unwrap_or((text, "0"));
    if !is_digits(whole) || !is_digits(fraction) {
        return Err("is not a decimal".to_string());
    }
    text.parse().map_err(|_| "is not a decimal".to_string())
}

/// Where the default of a knob comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fallback {
    Fixed(Value),
    /// The settings of the scenario derive it from its other knobs, see
    /// `Knobs::derive`.
    Derived,
}

/// A knob as a scenario declares it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knob {
    pub name: &'static str,
    pub class: Class,
    pub kind: Kind,
    pub default: Fallback,
    pub about: &'static str,
}

/// The name of a knob and the type its value reads as. Scenarios that share a knob
/// share its key, and each declares its own default and bounds.
pub struct Key<T> {
    pub name: &'static str,
    value: PhantomData<fn() -> T>,
}

impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Key<T> {}

impl<T> Key<T> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            value: PhantomData,
        }
    }

    const fn knob(self, class: Class, kind: Kind, default: Fallback, about: &'static str) -> Knob {
        Knob {
            name: self.name,
            class,
            kind,
            default,
            about,
        }
    }
}

impl Key<u64> {
    pub const fn count(
        self,
        class: Class,
        default: u64,
        bounds: RangeInclusive<u64>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Count {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        self.knob(class, kind, Fallback::Fixed(Value::Count(default)), about)
    }

    pub const fn bytes(
        self,
        class: Class,
        default: u64,
        bounds: RangeInclusive<u64>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Bytes {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        self.knob(class, kind, Fallback::Fixed(Value::Bytes(default)), about)
    }

    pub fn get(&self, knobs: &Knobs) -> u64 {
        match knobs.value(self.name) {
            Value::Count(amount) | Value::Bytes(amount) => amount,
            other => panic!("{} is not a count or a size: {other:?}", self.name),
        }
    }

    /// Sets it, for a run that decides it for itself.
    pub fn set(&self, knobs: &mut Knobs, amount: u64) {
        let value = match knobs.setting(self.name).knob.kind {
            Kind::Count { .. } => Value::Count(amount),
            Kind::Bytes { .. } => Value::Bytes(amount),
            other => panic!("{} is not a count or a size: {other:?}", self.name),
        };
        knobs.set(self.name, value);
    }
}

impl Key<RangeInclusive<u64>> {
    pub const fn range(
        self,
        class: Class,
        default: RangeInclusive<u64>,
        bounds: RangeInclusive<u64>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Range {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        let default = Value::Range(*default.start(), *default.end());
        self.knob(class, kind, Fallback::Fixed(default), about)
    }

    pub fn get(&self, knobs: &Knobs) -> RangeInclusive<u64> {
        match knobs.value(self.name) {
            Value::Range(start, end) => start..=end,
            other => panic!("{} is not a range: {other:?}", self.name),
        }
    }
}

impl Key<Duration> {
    pub const fn duration(
        self,
        class: Class,
        default: Duration,
        bounds: RangeInclusive<Duration>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Duration {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        self.knob(
            class,
            kind,
            Fallback::Fixed(Value::Duration(default)),
            about,
        )
    }

    /// A length whose default the settings of the scenario derive.
    pub const fn derived(
        self,
        class: Class,
        bounds: RangeInclusive<Duration>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Duration {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        self.knob(class, kind, Fallback::Derived, about)
    }

    pub fn get(&self, knobs: &Knobs) -> Duration {
        match knobs.value(self.name) {
            Value::Duration(length) => length,
            other => panic!("{} is not a length: {other:?}", self.name),
        }
    }

    /// Its value, where the engine serves scenarios with and without it.
    pub fn find(&self, knobs: &Knobs) -> Option<Duration> {
        knobs.declares(self.name).then(|| self.get(knobs))
    }

    /// Sets it, for a run that decides it for itself.
    pub fn set(&self, knobs: &mut Knobs, length: Duration) {
        knobs.set(self.name, Value::Duration(length));
    }

    /// The value a source set, if one did.
    pub fn explicit(&self, knobs: &Knobs) -> Option<Duration> {
        match knobs.setting(self.name).set {
            Some(Value::Duration(length)) => Some(length),
            _ => None,
        }
    }
}

impl Key<f64> {
    pub const fn share(
        self,
        class: Class,
        default: f64,
        bounds: RangeInclusive<f64>,
        about: &'static str,
    ) -> Knob {
        let kind = Kind::Share {
            least: *bounds.start(),
            most: *bounds.end(),
        };
        self.knob(class, kind, Fallback::Fixed(Value::Share(default)), about)
    }

    pub fn get(&self, knobs: &Knobs) -> f64 {
        match knobs.value(self.name) {
            Value::Share(share) => share,
            other => panic!("{} is not a share: {other:?}", self.name),
        }
    }
}

pub const SEEDS: Key<u64> = Key::new("seeds");
pub const SEED_BASE: Key<u64> = Key::new("seed_base");
pub const REPEAT: Key<u64> = Key::new("repeat");
pub const STALL: Key<Duration> = Key::new("stall");
pub const DEADLINE: Key<Duration> = Key::new("deadline");
pub const STEPS: Key<u64> = Key::new("steps");

/// The knobs of the parent that every scenario has: how many seeds a run takes,
/// `seeds` by default, and when the parent gives up on a child.
pub fn harness(seeds: u64) -> Vec<Knob> {
    vec![
        SEEDS.count(
            Class::Shape,
            seeds,
            1..=100_000,
            "how many seeds a run takes",
        ),
        SEED_BASE.count(Class::Shape, 1, 0..=u64::MAX, "the first seed a run takes"),
        REPEAT.count(Class::Shape, 1, 1..=1000, "how many times each seed runs"),
        STALL.derived(
            Class::Judge,
            Duration::from_secs(30)..=Duration::from_secs(86_400),
            "how long a child may go without progress",
        ),
        DEADLINE.derived(
            Class::Judge,
            Duration::from_secs(60)..=Duration::from_secs(30 * 86_400),
            "how long a child may run",
        ),
    ]
}

/// The steps of a plan: operations, or the rounds of `concurrency`.
pub const fn steps() -> Knob {
    STEPS.count(
        Class::Plan,
        8,
        1..=64,
        "operations per seed, or rounds of concurrency",
    )
}

/// Where a value of a source comes from, for its errors.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Origin {
    File { path: String, line: usize },
    Set,
    Short(&'static str),
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File { path, line } => write!(f, "{ENV_SET_FILE} {path} line {line}"),
            Self::Set => f.write_str(ENV_SET),
            Self::Short(variable) => f.write_str(variable),
        }
    }
}

impl Origin {
    /// What a variable sets wins over the file.
    fn layer(&self) -> u8 {
        match self {
            Self::File { .. } => 0,
            Self::Set | Self::Short(_) => 1,
        }
    }

    /// The source as a whole, without the line.
    fn source(&self) -> String {
        match self {
            Self::File { path, .. } => format!("{ENV_SET_FILE} {path}"),
            other => other.to_string(),
        }
    }
}

/// One `key=value` of a source.
#[derive(Clone, Debug)]
struct Item {
    /// A scenario of the registry, for a key with a prefix.
    scenario: Option<&'static str>,
    knob: String,
    text: String,
    origin: Origin,
}

impl Item {
    fn key(&self) -> String {
        match self.scenario {
            Some(scenario) => format!("{scenario}.{}", self.knob),
            None => self.knob.clone(),
        }
    }

    fn applies_to(&self, scenario: &dyn Scenario, knob: &Knob) -> bool {
        self.knob == knob.name && self.scenario.is_none_or(|name| name == scenario.name())
    }
}

/// Reads the pairs of one source, `<key>=<value>` each.
fn read_pairs(
    pairs: impl Iterator<Item = (Origin, String)>,
    items: &mut Vec<Item>,
) -> Result<(), String> {
    let first = items.len();
    for (origin, pair) in pairs {
        let (key, text) = pair
            .split_once('=')
            .map(|(key, text)| (key.trim(), text.trim()))
            .filter(|(key, text)| !key.is_empty() && !text.is_empty())
            .ok_or_else(|| format!("{origin}: {pair:?} is not `<knob>=<value>`"))?;
        let (scenario, knob) = match key.split_once('.') {
            Some((scenario, knob)) => {
                let known = REGISTRY
                    .iter()
                    .find(|known| known.name() == scenario)
                    .ok_or_else(|| {
                        format!(
                            "{origin}: {key:?} names no scenario; the scenarios are {}",
                            scenario_names()
                        )
                    })?;
                (Some(known.name()), knob)
            }
            None => (None, key),
        };
        let item = Item {
            scenario,
            knob: knob.to_string(),
            text: text.to_string(),
            origin,
        };
        if let Some(earlier) = items[first..]
            .iter()
            .find(|earlier| earlier.key() == item.key())
        {
            let lines = match (&earlier.origin, &item.origin) {
                (Origin::File { line: first, .. }, Origin::File { line, .. }) => {
                    format!(", on lines {first} and {line}")
                }
                _ => String::new(),
            };
            return Err(format!(
                "{} sets {} twice{lines}",
                item.origin.source(),
                item.key()
            ));
        }
        items.push(item);
    }
    Ok(())
}

/// Every `key=value` of every source in `env`: the file, `LANCE_STRESS_SET`, and
/// the short forms.
fn read_sources(env: &Env) -> Result<Vec<Item>, String> {
    let mut items = Vec::new();
    // Empty, as an input of CI left blank gives it, a variable sets nothing.
    if let Some(path) = env.get(ENV_SET_FILE).filter(|path| !path.trim().is_empty()) {
        let text = std::fs::read_to_string(path).map_err(|error| {
            let dir = std::env::current_dir()
                .map_or_else(|_| "?".to_string(), |dir| dir.display().to_string());
            format!("{ENV_SET_FILE}={path:?}, read from {dir}: {error}")
        })?;
        let lines = text.lines().enumerate().filter_map(|(at, line)| {
            let pair = line.split('#').next().unwrap_or("").trim();
            (!pair.is_empty()).then(|| {
                let origin = Origin::File {
                    path: path.to_string(),
                    line: at + 1,
                };
                (origin, pair.to_string())
            })
        });
        read_pairs(lines, &mut items)?;
    }
    if let Some(set) = env.get(ENV_SET).filter(|set| !set.trim().is_empty()) {
        read_pairs(
            set.split(',').map(|pair| (Origin::Set, pair.to_string())),
            &mut items,
        )?;
    }
    for (variable, scenario, knob, unit) in SHORT_FORMS {
        if let Some(text) = env.get(variable) {
            parse_count(text).map_err(|reason| {
                let declared = declared(scenario, knob).expect("a short form sets a knob");
                let unit = match unit {
                    "s" => " of seconds",
                    _ => "",
                };
                format!(
                    "{variable}={text:?} {reason}: it sets {knob} as a whole number{unit}, within {}",
                    declared.kind.bounds()
                )
            })?;
            items.push(Item {
                scenario,
                knob: knob.to_string(),
                text: format!("{text}{unit}"),
                origin: Origin::Short(variable),
            });
        }
    }
    Ok(items)
}

fn scenario_names() -> String {
    let names: Vec<&str> = REGISTRY.iter().map(|scenario| scenario.name()).collect();
    names.join(", ")
}

/// The knob `name` as the first scenario that has it declares it, of those
/// `scenario` names.
fn declared(scenario: Option<&str>, name: &str) -> Option<Knob> {
    REGISTRY
        .iter()
        .filter(|known| scenario.is_none_or(|scenario| scenario == known.name()))
        .find_map(|known| known.knobs().into_iter().find(|knob| knob.name == name))
}

/// A knob of a scenario and where its value comes from.
#[derive(Clone, Copy, Debug)]
struct Setting {
    knob: Knob,
    /// What a source set it to.
    set: Option<Value>,
    /// Its default, which the settings of the scenario derive when it has none
    /// of its own.
    default: Option<Value>,
}

impl Setting {
    fn value(&self) -> Option<Value> {
        self.set.or(self.default)
    }
}

/// The knobs of one scenario, each with its value.
#[derive(Clone, Debug)]
pub struct Knobs {
    scenario: &'static str,
    settings: BTreeMap<&'static str, Setting>,
    /// What set the deadline, for the error of one too short for what runs.
    deadline_source: Option<Item>,
}

impl Knobs {
    /// Every knob of `scenario` at its default.
    pub fn defaults(scenario: &dyn Scenario) -> Self {
        let settings = scenario
            .knobs()
            .into_iter()
            .map(|knob| {
                let default = match knob.default {
                    Fallback::Fixed(value) => Some(value),
                    Fallback::Derived => None,
                };
                let setting = Setting {
                    knob,
                    set: None,
                    default,
                };
                (knob.name, setting)
            })
            .collect();
        Self {
            scenario: scenario.name(),
            settings,
            deadline_source: None,
        }
    }

    /// The knobs of `scenario` as the sources of `env` set them. Every key of every
    /// source is checked against every scenario of the registry first.
    pub fn resolve(scenario: &dyn Scenario, env: &Env) -> Result<Self, String> {
        let items = read_sources(env)?;
        for item in &items {
            if declared(item.scenario, &item.knob).is_none() {
                let missing = match item.scenario {
                    Some(scenario) => format!("{scenario} has no knob"),
                    None => "no scenario has a knob".to_string(),
                };
                return Err(format!(
                    "{}: {missing} {:?}; `cargo test -p lance --test stress -- --exact knobs::list --nocapture` lists them",
                    item.origin, item.knob
                ));
            }
        }
        let mut resolved = None;
        for known in REGISTRY {
            let knobs = Self::resolve_for(*known, &items)?;
            if known.name() == scenario.name() {
                resolved = Some(knobs);
            }
        }
        Ok(resolved.unwrap_or_else(|| panic!("{} is missing from the registry", scenario.name())))
    }

    /// What the items set for one scenario.
    fn resolve_for(scenario: &dyn Scenario, items: &[Item]) -> Result<Self, String> {
        let mut knobs = Self::defaults(scenario);
        let mut winners = BTreeMap::new();
        for setting in knobs.settings.values_mut() {
            let knob = setting.knob;
            let applying: Vec<&Item> = items
                .iter()
                .filter(|item| item.applies_to(scenario, &knob))
                .collect();
            let short = applying
                .iter()
                .find(|item| matches!(item.origin, Origin::Short(_)));
            let set = applying.iter().find(|item| item.origin == Origin::Set);
            if let (Some(short), Some(set)) = (short, set) {
                return Err(format!(
                    "{} ({}={}) and {}={} in {ENV_SET} both set {}.{}: keep one",
                    short.origin,
                    short.knob,
                    short.text,
                    set.key(),
                    set.text,
                    scenario.name(),
                    knob.name
                ));
            }
            // A later layer wins, and within one a key with a prefix.
            let Some(winner) = applying
                .iter()
                .max_by_key(|item| (item.origin.layer(), item.scenario.is_some()))
            else {
                continue;
            };
            for item in &applying {
                let value = knob.kind.parse(&item.text).map_err(|reason| {
                    let mut error = format!(
                        "{}: {}={} {reason} for {}",
                        item.origin,
                        item.key(),
                        item.text,
                        scenario.name()
                    );
                    // A value that suits another scenario may be meant for it alone.
                    let suits_another = item.scenario.is_none()
                        && !matches!(item.origin, Origin::Short(_))
                        && REGISTRY.iter().any(|other| {
                            other.knobs().iter().any(|other| {
                                other.name == knob.name && other.kind.parse(&item.text).is_ok()
                            })
                        });
                    if suits_another {
                        error.push_str(&format!(
                            "; a key without a prefix sets the knob of every scenario that has it, so set it for one with `<scenario>.{}`",
                            knob.name
                        ));
                    }
                    error
                })?;
                if std::ptr::eq(*item, *winner) {
                    setting.set = Some(value);
                    winners.insert(knob.name, *winner);
                }
            }
        }
        let base = SEED_BASE.get(&knobs);
        let seeds = SEEDS.get(&knobs);
        if base.checked_add(seeds).is_none() {
            let from = |name: &str| {
                winners
                    .get(name)
                    .map_or_else(|| "its default".to_string(), |item| item.origin.to_string())
            };
            return Err(format!(
                "{}: seed_base={base} from {} and seeds={seeds} from {} run past the last seed",
                scenario.name(),
                from(SEED_BASE.name),
                from(SEEDS.name)
            ));
        }
        knobs.deadline_source = winners.get(DEADLINE.name).map(|item| (*item).clone());
        Ok(knobs)
    }

    /// A deadline set by hand must leave what runs the least time its scenario
    /// declares, which only a run of `soak` does: its length and a minute. It is a
    /// matter of the test that runs, so a deadline that suits the others does not
    /// fail them.
    pub fn check_least_deadline(
        &self,
        scenario: &dyn Scenario,
        self_test: Option<&SelfTest>,
    ) -> Result<(), String> {
        let (Some(item), Some((least, why))) = (
            &self.deadline_source,
            scenario.settings(self_test, self).least_deadline,
        ) else {
            return Ok(());
        };
        if DEADLINE.get(self) >= least {
            return Ok(());
        }
        let mut error = format!(
            "{}: {}={} is shorter than the {} that {} needs: {why}",
            item.origin,
            item.key(),
            item.text,
            Value::Duration(least),
            scenario.name()
        );
        if item.scenario.is_none() {
            error.push_str(&format!(
                "; it sets the deadline of every scenario, so set the one of a scenario with `<scenario>.{}` in {ENV_SET}",
                DEADLINE.name
            ));
        }
        Err(error)
    }

    pub fn declares(&self, name: &str) -> bool {
        self.settings.contains_key(name)
    }

    fn setting(&self, name: &str) -> &Setting {
        self.settings
            .get(name)
            .unwrap_or_else(|| panic!("{} declares no knob {name}", self.scenario))
    }

    fn value(&self, name: &str) -> Value {
        self.setting(name).value().unwrap_or_else(|| {
            panic!(
                "{}.{name} has no value until the settings of the scenario derive it",
                self.scenario
            )
        })
    }

    /// The setting of `name`, for a value the code gives it, which must read back
    /// from its record within the bounds, or a child could not take it.
    fn setting_for(&mut self, name: &str, value: Value) -> &mut Setting {
        let scenario = self.scenario;
        let setting = self
            .settings
            .get_mut(name)
            .unwrap_or_else(|| panic!("{scenario} declares no knob {name}"));
        assert_eq!(
            setting.knob.kind.parse(&value.to_string()),
            Ok(value),
            "{scenario}.{name}: {value:?} does not read back from its record"
        );
        setting
    }

    fn set(&mut self, name: &str, value: Value) {
        self.setting_for(name, value).set = Some(value);
    }

    /// Gives a knob of `Fallback::Derived` its default, which the settings of the
    /// scenario derive from the other knobs.
    pub fn derive(&mut self, key: Key<Duration>, default: Duration) {
        let setting = self.setting_for(key.name, Value::Duration(default));
        assert_eq!(setting.knob.default, Fallback::Derived, "{}", key.name);
        setting.default = Some(Value::Duration(default));
    }

    /// Every knob of the classes plan and judge, as `<scenario>.<knob>=<value>`
    /// separated by commas: what a child runs with and what a replay pins. A
    /// derived one goes in only when a source set it: a replay derives it again,
    /// from knobs that its user may have changed.
    pub fn record(&self) -> String {
        let pairs: Vec<String> = self
            .settings
            .values()
            .filter(|setting| setting.knob.class != Class::Shape)
            .filter(|setting| setting.knob.default != Fallback::Derived || setting.set.is_some())
            .map(|setting| {
                format!(
                    "{}.{}={}",
                    self.scenario,
                    setting.knob.name,
                    self.value(setting.knob.name)
                )
            })
            .collect();
        pairs.join(",")
    }

    /// The knobs of `class` off their default, but those in `shown`, as
    /// `<scenario>.<knob>=<value>`.
    pub fn off_default(&self, class: Class, shown: &[&str]) -> Vec<String> {
        self.settings
            .values()
            .filter(|setting| setting.knob.class == class && !shown.contains(&setting.knob.name))
            .filter(|setting| setting.value() != setting.default)
            .map(|setting| {
                format!(
                    "{}.{}={}",
                    self.scenario,
                    setting.knob.name,
                    self.value(setting.knob.name)
                )
            })
            .collect()
    }
}

#[cfg(test)]
fn named(name: &str) -> &'static dyn Scenario {
    crate::scenario::by_name(name).expect("a scenario of the registry")
}

#[cfg(test)]
fn resolve(name: &str, vars: &[(&str, &str)]) -> Result<Knobs, String> {
    let env = Env(vars
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect());
    Knobs::resolve(named(name), &env)
}

#[test]
fn values_read_and_print_in_their_units() {
    let length = |text: &str| {
        let kind = Kind::Duration {
            least: Duration::ZERO,
            most: Duration::MAX,
        };
        kind.parse(text)
    };
    assert_eq!(
        length("7.5s"),
        Ok(Value::Duration(Duration::from_millis(7500)))
    );
    assert_eq!(
        length("500ms"),
        Ok(Value::Duration(Duration::from_millis(500)))
    );
    assert_eq!(length("2m"), Ok(Value::Duration(Duration::from_secs(120))));
    assert_eq!(
        length("1d"),
        Ok(Value::Duration(Duration::from_secs(86_400)))
    );
    // Zeros at the end of a fraction add nothing, however many.
    assert_eq!(
        length("1.5000000000000d"),
        Ok(Value::Duration(Duration::from_secs(129_600)))
    );
    assert_eq!(
        length("1.00000000000000000000s"),
        Ok(Value::Duration(Duration::from_secs(1)))
    );
    for (bad, reason) in [
        ("180", "without a unit"),
        ("1.0001s", "finer than its smallest unit, ms"),
        ("s", "is not a length"),
        ("1.s", "is not a length"),
        (".5s", "is not a length"),
        ("1..2s", "is not a length"),
        ("-1s", "is not a length"),
        ("1 s", "is not a length"),
        ("1e3ms", "is not a length"),
        ("99999999999999999999s", "is too large"),
        ("18446744073709551615d", "is too large"),
    ] {
        let error = length(bad).unwrap_err();
        assert!(error.contains(reason), "{bad:?}: {error}");
    }
    let size = |text: &str| {
        let kind = Kind::Bytes {
            least: 0,
            most: u64::MAX,
        };
        kind.parse(text)
    };
    assert_eq!(size("1.5KiB"), Ok(Value::Bytes(1536)));
    assert_eq!(size("16KiB"), Ok(Value::Bytes(16 << 10)));
    for bad in ["4096", "0.1B", "1KB", "99999999999GiB"] {
        assert!(size(bad).is_err(), "{bad:?}");
    }
    let share = Kind::Share {
        least: 0.0,
        most: 10.0,
    };
    // As the literal of the same digits, so a default does not move a draw.
    assert_eq!(share.parse("0.7"), Ok(Value::Share(0.7)));
    assert_eq!(
        share.parse("0.700000000000000000000"),
        Ok(Value::Share(0.7))
    );
    assert_eq!(share.parse("1"), Ok(Value::Share(1.0)));
    for bad in ["", ".7", "7.", "1e-3", "-0.5", "inf", "NaN", "0x1"] {
        assert!(share.parse(bad).is_err(), "{bad:?}");
    }
    let error = share.parse("100000000000000000000").unwrap_err();
    assert!(error.contains("out of its bounds 0..=10"), "{error}");
    let range = Kind::Range { least: 1, most: 9 };
    assert_eq!(range.parse("2..=4"), Ok(Value::Range(2, 4)));
    assert_eq!(range.parse("3"), Ok(Value::Range(3, 3)));
    // Whatever is wrong with a value, the error names the bounds.
    for bad in ["2..4", "4..=2", "0..=4", "2..=10", "..=4", "two"] {
        let error = range.parse(bad).unwrap_err();
        assert!(error.contains("1..=9"), "{bad:?}: {error}");
    }
    // Each prints in its largest exact unit, and reads back as itself.
    for (value, text) in [
        (Value::Duration(Duration::from_millis(7500)), "7500ms"),
        (Value::Duration(Duration::from_secs(180)), "3m"),
        (Value::Duration(Duration::from_secs(150)), "150s"),
        (Value::Duration(Duration::from_secs(86_400)), "1d"),
        (Value::Bytes(3 << 20), "3MiB"),
        (Value::Bytes(1536), "1536B"),
        (Value::Bytes(0), "0B"),
        (Value::Share(0.35), "0.35"),
        (Value::Share(1.0), "1"),
        (Value::Range(2, 4), "2..=4"),
        (Value::Count(0), "0"),
    ] {
        assert_eq!(value.to_string(), text);
        let kind = match value {
            Value::Count(_) => Kind::Count {
                least: 0,
                most: u64::MAX,
            },
            Value::Range(..) => Kind::Range {
                least: 0,
                most: u64::MAX,
            },
            Value::Duration(_) => Kind::Duration {
                least: Duration::ZERO,
                most: Duration::MAX,
            },
            Value::Bytes(_) => Kind::Bytes {
                least: 0,
                most: u64::MAX,
            },
            Value::Share(_) => share,
        };
        assert_eq!(kind.parse(text), Ok(value), "{text}");
    }
}

#[test]
fn defaults_hold_the_numbers_of_the_code() {
    let knobs = Knobs::defaults(named("storage_faults"));
    assert_eq!(SEEDS.get(&knobs), 16);
    assert_eq!(STEPS.get(&knobs), 8);
    let soak = Knobs::defaults(named("soak"));
    assert_eq!(SEEDS.get(&soak), 1);
    assert!(!soak.declares(STEPS.name));
    // With no source, no knob counts as set: the stall and the deadline stay
    // derived, and the record leaves them to the replay.
    for scenario in REGISTRY {
        let knobs = resolve(scenario.name(), &[]).unwrap();
        assert!(
            knobs.settings.values().all(|setting| setting.set.is_none()),
            "{}",
            scenario.name()
        );
        assert!(!knobs.record().contains("stall="), "{}", knobs.record());
    }
}

#[test]
fn layers_go_file_then_variables_and_a_prefix_wins_within_one() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("knobs.txt");
    std::fs::write(
        &file,
        "# the file\nconcurrency.seeds=100 # a comment\nsteps=4\n\nrepeat=2\n",
    )
    .unwrap();
    let path = file.to_str().unwrap();
    // The variable is a later layer than the file, whatever their prefixes.
    let knobs = resolve(
        "concurrency",
        &[
            (ENV_SET_FILE, path),
            (ENV_SET, "seeds=4,concurrency.steps=6"),
        ],
    )
    .unwrap();
    assert_eq!(SEEDS.get(&knobs), 4);
    assert_eq!(STEPS.get(&knobs), 6);
    assert_eq!(REPEAT.get(&knobs), 2);
    // Within one layer a key with a prefix wins.
    let knobs = resolve(
        "concurrency",
        &[
            (ENV_SET, "steps=4,concurrency.steps=6"),
            (ENV_SET_FILE, path),
        ],
    )
    .unwrap();
    assert_eq!(STEPS.get(&knobs), 6);
    // In whichever order the two come.
    let knobs = resolve("concurrency", &[(ENV_SET, "concurrency.steps=6,steps=4")]).unwrap();
    assert_eq!(STEPS.get(&knobs), 6);
    let knobs = resolve(
        "storage_faults",
        &[(ENV_SET, "steps=4,concurrency.steps=6")],
    )
    .unwrap();
    assert_eq!(STEPS.get(&knobs), 4);
    let knobs = resolve("concurrency", &[(ENV_SET_FILE, path)]).unwrap();
    assert_eq!((SEEDS.get(&knobs), STEPS.get(&knobs)), (100, 4));
    let knobs = resolve("storage_faults", &[(ENV_SET_FILE, path)]).unwrap();
    assert_eq!((SEEDS.get(&knobs), STEPS.get(&knobs)), (16, 4));
    let prefixed_first = dir.path().join("prefixed_first.txt");
    std::fs::write(&prefixed_first, "concurrency.steps=6\nsteps=4\n").unwrap();
    let prefixed_first = prefixed_first.to_str().unwrap();
    for (name, steps) in [("concurrency", 6), ("storage_faults", 4)] {
        let knobs = resolve(name, &[(ENV_SET_FILE, prefixed_first)]).unwrap();
        assert_eq!(STEPS.get(&knobs), steps, "{name}");
    }
    // A short form is a variable too, and wins over the file with a prefix.
    let knobs = resolve(
        "concurrency",
        &[(ENV_SET_FILE, prefixed_first), (ENV_STEPS, "5")],
    )
    .unwrap();
    assert_eq!(STEPS.get(&knobs), 5);
    // A value that loses is checked all the same.
    let wrong = dir.path().join("wrong.txt");
    std::fs::write(&wrong, "steps=0\n").unwrap();
    let error = resolve(
        "concurrency",
        &[
            (ENV_SET_FILE, wrong.to_str().unwrap()),
            (ENV_SET, "steps=4"),
        ],
    )
    .unwrap_err();
    assert!(
        error.contains("line 1: steps=0 is out of its bounds"),
        "{error}"
    );
    // Empty, as a blank input of CI leaves it, the variable names no file.
    let knobs = resolve("concurrency", &[(ENV_SET_FILE, ""), (ENV_SET, "")]).unwrap();
    assert_eq!(STEPS.get(&knobs), 8);
    // A relative path is the process's, which the error names.
    let error = resolve("storage_faults", &[(ENV_SET_FILE, "no/such/knobs.txt")]).unwrap_err();
    assert!(
        error.contains("no/such/knobs.txt") && error.contains("read from /"),
        "{error}"
    );
}

#[test]
fn short_forms_read_as_before_and_conflict_with_the_set() {
    let knobs = resolve(
        "soak",
        &[
            ("LANCE_STRESS_SEEDS", "3"),
            ("LANCE_STRESS_SEED_BASE", "7"),
            ("LANCE_STRESS_STALL_SECS", "200"),
            (ENV_SOAK_SECS, "4321"),
        ],
    )
    .unwrap();
    assert_eq!((SEEDS.get(&knobs), SEED_BASE.get(&knobs)), (3, 7));
    assert_eq!(STALL.explicit(&knobs), Some(Duration::from_secs(200)));
    assert_eq!(knobs.off_default(Class::Plan, &[]), ["soak.secs=4321s"]);
    // One knob from two variables of the same layer is a mistake of the shell.
    for (short, set) in [
        ((ENV_STEPS, "4"), "steps=6"),
        ((ENV_STEPS, "4"), "concurrency.steps=6"),
        ((ENV_SOAK_SECS, "60"), "soak.secs=2m"),
        (("LANCE_STRESS_DEADLINE_SECS", "900"), "deadline=20m"),
    ] {
        let error = resolve("storage_faults", &[short, (ENV_SET, set)]).unwrap_err();
        assert!(
            error.contains(short.0)
                && error.contains(&format!("={}", short.1))
                && error.contains(set),
            "{error}"
        );
    }
    // As before, a short form takes an integer in its old unit; the error names the
    // knob it sets and the bounds.
    for (bad, expected) in [
        (
            ("LANCE_STRESS_STALL_SECS", "60s"),
            "it sets stall as a whole number of seconds, within 30s..=1d",
        ),
        (
            ("LANCE_STRESS_SEEDS", "-1"),
            "it sets seeds as a whole number, within 1..=100000",
        ),
    ] {
        let error = resolve("storage_faults", &[bad]).unwrap_err();
        assert!(error.contains(bad.0) && error.contains(expected), "{error}");
    }
    // A short form takes no prefix, so its error suggests none.
    let error = resolve("storage_faults", &[(ENV_STEPS, "0")]).unwrap_err();
    assert!(
        error.contains("steps=0 is out of its bounds 1..=64") && !error.contains("<scenario>"),
        "{error}"
    );
}

#[test]
fn every_key_is_checked_against_the_whole_registry() {
    for (set, expected) in [
        ("stepz=4", "no scenario has a knob \"stepz\""),
        ("soak.steps=4", "soak has no knob \"steps\""),
        ("faults.steps=4", "names no scenario"),
        ("steps=0", "out of its bounds 1..=64"),
        ("steps=4,steps=5", "sets steps twice"),
        ("steps", "is not `<knob>=<value>`"),
        ("steps=4,", "is not `<knob>=<value>`"),
        ("deadline=10", "without a unit"),
        ("repeat=0", "out of its bounds 1..=1000"),
        ("seed_base=18446744073709551615", "run past the last seed"),
        ("stall=29s", "out of its bounds 30s..=1d"),
    ] {
        // The test of a scenario the key does not concern fails as well.
        let error = resolve("storage_faults", &[(ENV_SET, set)]).unwrap_err();
        assert!(error.contains(expected), "{set}: {error}");
    }
    let error = resolve("storage_faults", &[(ENV_SET, "concurrency.steps=65")]).unwrap_err();
    assert!(
        error.contains("concurrency.steps=65 is out of its bounds"),
        "{error}"
    );
    // A key without a prefix is told to take one only where that would help.
    for (set, helps) in [
        ("workers=2..=8", true),
        ("steps=0", false),
        ("deadline=10", false),
    ] {
        let error = resolve("storage_faults", &[(ENV_SET, set)]).unwrap_err();
        assert_eq!(error.contains("<scenario>."), helps, "{set}: {error}");
    }
}

/// A child reads only its record, and must land on the knobs of its parent: every
/// one at its default, and every one off it.
/// The least deadline is that of the test that runs: a short form, which takes no
/// prefix, sets the deadline of every scenario, and only a run of `soak` asks for
/// more than a minute.
#[test]
fn a_deadline_is_held_to_the_least_one_of_what_runs() {
    let soak = named("soak");
    let test = soak.self_tests()[0];
    let short = [("LANCE_STRESS_DEADLINE_SECS", "200")];
    let knobs = resolve("storage_faults", &short).unwrap();
    knobs
        .check_least_deadline(named("storage_faults"), None)
        .unwrap();
    let knobs = resolve("soak", &short).unwrap();
    knobs.check_least_deadline(soak, Some(test)).unwrap();
    let error = knobs.check_least_deadline(soak, None).unwrap_err();
    assert!(
        error.starts_with(
            "LANCE_STRESS_DEADLINE_SECS: deadline=200s is shorter than the 4m that soak needs"
        ) && error.contains("`<scenario>.deadline`"),
        "{error}"
    );
    // A key with a prefix is told nothing about prefixes.
    let knobs = resolve("soak", &[(ENV_SET, "soak.deadline=2m")]).unwrap();
    let error = knobs.check_least_deadline(soak, None).unwrap_err();
    assert!(
        error.contains("soak.deadline=2m is shorter") && !error.contains("<scenario>"),
        "{error}"
    );
    // A derived deadline is not held: it is no one's mistake.
    let knobs = resolve("soak", &[(ENV_SET, "soak.secs=1h")]).unwrap();
    knobs.check_least_deadline(soak, None).unwrap();
}

/// The floors that keep a knob out of the values that break the meaning of its
/// number, each with the value just below: one that drops back fails here.
#[test]
fn raised_floors_hold() {
    let mut cases = vec![
        ("concurrency", "concurrency.gate_limit", "9999ms", "10s"),
        ("slow_storage", "slow_storage.late_after", "49ms", "50ms"),
        ("slow_storage", "slow_storage.settle", "1999ms", "2s"),
        ("slow_storage", "slow_storage.delay_scale", "0.39", "0.4"),
        ("heap_growth", "heap_growth.limit", "4095B", "4KiB"),
        ("soak", "soak.interval", "7499ms", "7500ms"),
        ("soak", "soak.secs", "86400001ms", "1d"),
    ];
    // A tolerance of `soak` only rises from its default.
    let below: Vec<(String, String, String)> = named("soak")
        .knobs()
        .iter()
        .filter(|knob| knob.name.ends_with("_tolerance"))
        .map(|knob| {
            let Fallback::Fixed(default) = knob.default else {
                panic!("{} has no default", knob.name)
            };
            let less = match default {
                Value::Bytes(bytes) => Value::Bytes(bytes - 1),
                Value::Count(count) => Value::Count(count - 1),
                other => panic!("{}: {other:?}", knob.name),
            };
            (
                format!("soak.{}", knob.name),
                less.to_string(),
                default.to_string(),
            )
        })
        .collect();
    assert_eq!(below.len(), 11, "{below:?}");
    cases.extend(
        below
            .iter()
            .map(|(key, less, least)| ("soak", key.as_str(), less.as_str(), least.as_str())),
    );
    for (scenario, key, refused, taken) in cases {
        let set = format!("{key}={refused}");
        let error = resolve(scenario, &[(ENV_SET, set.as_str())]).unwrap_err();
        assert!(error.contains("out of its bounds"), "{set}: {error}");
        let set = format!("{key}={taken}");
        resolve(scenario, &[(ENV_SET, set.as_str())])
            .unwrap_or_else(|error| panic!("{set}: {error}"));
    }
}

/// A variable of Lance that is not UTF-8 could not go into a replay line; another
/// one is left out.
#[test]
#[should_panic(expected = "LANCE_IO_THREADS is not UTF-8")]
fn the_environment_refuses_a_variable_of_lance_that_is_not_utf8() {
    use std::os::unix::ffi::OsStringExt;
    let bad = || OsString::from_vec(vec![b'4', 0xff]);
    let env = Env::of([
        (OsString::from("OTHER"), bad()),
        (OsString::from("LANCE_IO_THREADS_X"), OsString::from("4")),
    ]);
    assert_eq!(env.get("OTHER"), None);
    assert_eq!(env.get("LANCE_IO_THREADS_X"), Some("4"));
    Env::of([(OsString::from("LANCE_IO_THREADS"), bad())]);
}

#[test]
fn the_record_reads_back_as_itself() {
    for scenario in REGISTRY {
        let defaults = Knobs::defaults(*scenario);
        // Every knob of the plan and the verdicts at whichever end of its bounds is
        // further from its default.
        let ends: Vec<String> = scenario
            .knobs()
            .iter()
            .filter(|knob| knob.class != Class::Shape)
            .map(|knob| {
                let (least, most) = match knob.kind {
                    Kind::Count { least, most } => (Value::Count(least), Value::Count(most)),
                    Kind::Range { least, most } => {
                        (Value::Range(least, least), Value::Range(least, most))
                    }
                    Kind::Duration { least, most } => {
                        (Value::Duration(least), Value::Duration(most))
                    }
                    Kind::Bytes { least, most } => (Value::Bytes(least), Value::Bytes(most)),
                    Kind::Share { least, most } => (Value::Share(least), Value::Share(most)),
                };
                let end = match knob.default == Fallback::Fixed(most) {
                    true => least,
                    false => most,
                };
                format!("{}.{}={end}", scenario.name(), knob.name)
            })
            .collect();
        let off = Knobs::resolve(*scenario, &Env::from([(ENV_SET, ends.join(",").as_str())]))
            .unwrap_or_else(|error| panic!("{error}"));
        for parent in [defaults, off] {
            let record = parent.record();
            assert!(
                record
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./=:,".contains(c)),
                "{record}"
            );
            assert!(!record.contains("seeds="), "{record}");
            let child = Knobs::resolve(*scenario, &Env::from([(ENV_SET, record.as_str())]))
                .unwrap_or_else(|error| panic!("{record}: {error}"));
            for (name, setting) in &parent.settings {
                let expected = match setting.knob.class {
                    Class::Shape => child.settings[name].default,
                    _ => setting.value(),
                };
                assert_eq!(child.settings[name].value(), expected, "{name}: {record}");
            }
        }
    }
}

/// The table of knobs, by scenario: `cargo test -p lance --test stress -- --exact
/// knobs::list --nocapture`.
#[test]
#[allow(clippy::print_stderr)]
fn list() {
    for scenario in REGISTRY {
        eprintln!("{}:", scenario.name());
        for knob in scenario.knobs() {
            let default = match knob.default {
                Fallback::Fixed(value) => value.to_string(),
                Fallback::Derived => "derived".to_string(),
            };
            eprintln!(
                "  {}.{} {} default={default} bounds={}: {}",
                scenario.name(),
                knob.name,
                knob.class.name(),
                knob.kind.bounds(),
                knob.about
            );
        }
    }
}

#[test]
fn declarations_hold_together() {
    let mut kinds: BTreeMap<&str, (Class, std::mem::Discriminant<Kind>)> = BTreeMap::new();
    for scenario in REGISTRY {
        let knobs = scenario.knobs();
        for (at, knob) in knobs.iter().enumerate() {
            let what = format!("{}.{}", scenario.name(), knob.name);
            assert!(
                knobs[..at].iter().all(|other| other.name != knob.name),
                "{what} is declared twice"
            );
            assert!(!knob.about.is_empty(), "{what} says nothing of itself");
            assert!(
                knob.name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
                "{what}"
            );
            if let Fallback::Fixed(value) = knob.default {
                // The default reads back from its text within the bounds.
                assert_eq!(knob.kind.parse(&value.to_string()), Ok(value), "{what}");
            }
            // A knob of one name means one thing in every scenario.
            let kind = (knob.class, std::mem::discriminant(&knob.kind));
            assert_eq!(*kinds.entry(knob.name).or_insert(kind), kind, "{what}");
        }
    }
}

#[test]
fn derived_defaults_hold_the_numbers_of_the_code_and_only_rise() {
    let secs = Duration::from_secs;
    for (name, stall, deadline) in [
        ("storage_faults", 60, 600),
        ("cancellation", 60, 600),
        ("process_kill", 60, 600),
        ("memory_limits", 60, 600),
        ("heap_growth", 60, 120 + 60 * 144),
        ("concurrency", 150, 300 + 120 * 8),
        ("slow_storage", 60, 600),
        ("soak", 150, 180 + 300),
    ] {
        let settings = named(name).settings(None, &Knobs::defaults(named(name)));
        assert_eq!(
            (settings.stall, settings.deadline),
            (secs(stall), secs(deadline)),
            "{name}"
        );
    }
    let settings = |name: &str, set: &str| {
        let knobs = resolve(name, &[(ENV_SET, set)]).unwrap();
        named(name).settings(None, &knobs)
    };
    // 8 slowed calls of 15 s each, a settle of 2 s and 20 s of quiet: 142 s of
    // silence.
    let slow = settings(
        "slow_storage",
        "slow_storage.delay_scale=5,slow_storage.slowed_calls=1..=8",
    );
    assert_eq!(
        (slow.stall, slow.deadline),
        (secs(152), secs(120 + 8 * 142))
    );
    // Shorter delays leave the numbers of the code.
    let slow = settings("slow_storage", "slow_storage.delay_scale=0.4");
    assert_eq!((slow.stall, slow.deadline), (secs(60), secs(600)));
    // A hold that times out waits the whole limit, which sets the unit of Lance's
    // backoff: both the silence and a round grow with it.
    let gate = settings("concurrency", "concurrency.gate_limit=30s");
    assert_eq!(
        (gate.stall, gate.deadline),
        (secs(450), secs(300 + 360 * 8))
    );
    let gate = settings("concurrency", "concurrency.gate_limit=12s");
    assert_eq!(
        (gate.stall, gate.deadline),
        (secs(180), secs(300 + 144 * 8))
    );
    let rounds = settings("concurrency", "steps=12");
    assert_eq!(rounds.deadline, secs(300 + 120 * 12));
    let crowd = settings("concurrency", "concurrency.workers=5..=8");
    assert_eq!(
        (crowd.stall, crowd.deadline),
        (secs(150), secs(300 + 240 * 8))
    );
    let cycles = settings("heap_growth", "heap_growth.cycles=32");
    assert_eq!((cycles.steps, cycles.deadline), (288, secs(120 + 60 * 288)));
    let long = settings("soak", "soak.secs=1h");
    assert_eq!(long.deadline, secs(3600 + 300));
    assert_eq!(
        long.least_deadline.map(|(least, _)| least),
        Some(secs(3660))
    );
    // A runtime has a thread for each worker, and 4 at least.
    let wide = resolve("concurrency", &[(ENV_SET, "concurrency.workers=2..=8")]).unwrap();
    assert_eq!(named("concurrency").runtime(&wide).worker_threads, 8);
    let soak = Knobs::defaults(named("soak"));
    assert_eq!(named("soak").runtime(&soak).worker_threads, 4);
}

/// Every knob of every scenario at its default, the stall and the deadline as the
/// settings derive them: a default that changes fails here.
#[test]
fn defaults_keep_their_record() {
    let expected = GOLDEN_RECORDS;
    let mut actual = Vec::new();
    for scenario in REGISTRY {
        let mut knobs = Knobs::defaults(*scenario);
        let settings = scenario.settings(None, &knobs);
        knobs.derive(STALL, settings.stall);
        knobs.derive(DEADLINE, settings.deadline);
        let pairs: Vec<String> = knobs
            .settings
            .values()
            .map(|setting| format!("{}={}", setting.knob.name, knobs.value(setting.knob.name)))
            .collect();
        actual.push(format!("{} {}", scenario.name(), pairs.join(",")));
    }
    assert_eq!(actual, expected, "{actual:#?}");
}

#[cfg(test)]
const GOLDEN_RECORDS: &[&str] = &[
    "storage_faults deadline=10m,repeat=1,seed_base=1,seeds=16,stall=1m,steps=8,trouble=0.75",
    "cancellation deadline=10m,repeat=1,seed_base=1,seeds=16,stall=1m,steps=8,trouble=0.85",
    "process_kill deadline=10m,repeat=1,seed_base=1,seeds=16,stall=1m,steps=8,trouble=0.85",
    "memory_limits budget_scale=1,deadline=10m,repeat=1,seed_base=1,seeds=16,stall=1m,steps=8,trouble=0.85",
    "heap_growth cycles=16,deadline=146m,limit=16KiB,repeat=1,seed_base=1,seeds=16,stall=1m",
    "concurrency catch_up=0.7,deadline=21m,gate_limit=10s,hold=0.6,repeat=1,seed_base=1,seeds=16,stall=150s,steps=8,workers=2..=4",
    "slow_storage deadline=10m,delay_scale=1,hang=0.35,late_after=50ms,repeat=1,seed_base=1,seeds=16,settle=2s,slowed_calls=1..=3,stall=1m,steps=8,trouble=0.8",
    "soak dataset_bytes_tolerance=2MiB,deadline=8m,fds_tolerance=4,files_tolerance=100,fragments_tolerance=24,heap_bytes_tolerance=2MiB,index_segments_tolerance=8,interval=7500ms,malloc_bytes_tolerance=3MiB,manifest_bytes_tolerance=16KiB,repeat=1,secs=3m,seed_base=1,seeds=1,stall=150s,tasks_tolerance=16,threads_tolerance=8,versions_tolerance=16,workers=2..=4",
];
