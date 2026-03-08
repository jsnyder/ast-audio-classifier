"""AudioSet label definitions and grouping for Home Assistant entities.

Maps the 527 AudioSet ontology labels to meaningful groups that become
HA binary_sensor entities (e.g., dog_bark, cat_meow, smoke_alarm).

Label list sourced from MIT/ast-finetuned-audioset-10-10-0.4593 config.json id2label.
"""

from __future__ import annotations

# Full AudioSet label list (527 labels, index-ordered).
# Source: MIT/ast-finetuned-audioset-10-10-0.4593 model config.json
# This must match the model's output order exactly.
_AUDIOSET_LABELS: list[str] = [
    # 0-9
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
    "Shout",
    "Bellow",
    # 10-19
    "Whoop",
    "Yell",
    "Battle cry",
    "Children shouting",
    "Screaming",
    "Whispering",
    "Laughter",
    "Baby laughter",
    "Giggle",
    "Snicker",
    # 20-29
    "Belly laugh",
    "Chuckle, chortle",
    "Crying, sobbing",
    "Baby cry, infant cry",
    "Whimper",
    "Wail, moan",
    "Sigh",
    "Singing",
    "Choir",
    "Yodeling",
    # 30-39
    "Chant",
    "Mantra",
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
    "Rapping",
    "Humming",
    "Groan",
    "Grunt",
    # 40-49
    "Whistling",
    "Breathing",
    "Wheeze",
    "Snoring",
    "Gasp",
    "Pant",
    "Snort",
    "Cough",
    "Throat clearing",
    "Sneeze",
    # 50-59
    "Sniff",
    "Run",
    "Shuffle",
    "Walk, footsteps",
    "Chewing, mastication",
    "Biting",
    "Gargling",
    "Stomach rumble",
    "Burping, eructation",
    "Hiccup",
    # 60-69
    "Fart",
    "Hands",
    "Finger snapping",
    "Clapping",
    "Heart sounds, heartbeat",
    "Heart murmur",
    "Cheering",
    "Applause",
    "Chatter",
    "Crowd",
    # 70-79
    "Hubbub, speech noise, speech babble",
    "Children playing",
    "Animal",
    "Domestic animals, pets",
    "Dog",
    "Bark",
    "Yip",
    "Howl",
    "Bow-wow",
    "Growling",
    # 80-89
    "Whimper (dog)",
    "Cat",
    "Purr",
    "Meow",
    "Hiss",
    "Caterwaul",
    "Livestock, farm animals, working animals",
    "Horse",
    "Clip-clop",
    "Neigh, whinny",
    # 90-99
    "Cattle, bovinae",
    "Moo",
    "Cowbell",
    "Pig",
    "Oink",
    "Goat",
    "Bleat",
    "Sheep",
    "Fowl",
    "Chicken, rooster",
    # 100-109
    "Cluck",
    "Crowing, cock-a-doodle-doo",
    "Turkey",
    "Gobble",
    "Duck",
    "Quack",
    "Goose",
    "Honk",
    "Wild animals",
    "Roaring cats (lions, tigers)",
    # 110-119
    "Roar",
    "Bird",
    "Bird vocalization, bird call, bird song",
    "Chirp, tweet",
    "Squawk",
    "Pigeon, dove",
    "Coo",
    "Crow",
    "Caw",
    "Owl",
    # 120-129
    "Hoot",
    "Bird flight, flapping wings",
    "Canidae, dogs, wolves",
    "Rodents, rats, mice",
    "Mouse",
    "Patter",
    "Insect",
    "Cricket",
    "Mosquito",
    "Fly, housefly",
    # 130-139
    "Buzz",
    "Bee, wasp, etc.",
    "Frog",
    "Croak",
    "Snake",
    "Rattle",
    "Whale vocalization",
    "Music",
    "Musical instrument",
    "Plucked string instrument",
    # 140-149
    "Guitar",
    "Electric guitar",
    "Bass guitar",
    "Acoustic guitar",
    "Steel guitar, slide guitar",
    "Tapping (guitar technique)",
    "Strum",
    "Banjo",
    "Sitar",
    "Mandolin",
    # 150-159
    "Zither",
    "Ukulele",
    "Keyboard (musical)",
    "Piano",
    "Electric piano",
    "Organ",
    "Electronic organ",
    "Hammond organ",
    "Synthesizer",
    "Sampler",
    # 160-169
    "Harpsichord",
    "Percussion",
    "Drum kit",
    "Drum machine",
    "Drum",
    "Snare drum",
    "Rimshot",
    "Drum roll",
    "Bass drum",
    "Timpani",
    # 170-179
    "Tabla",
    "Cymbal",
    "Hi-hat",
    "Wood block",
    "Tambourine",
    "Rattle (instrument)",
    "Maraca",
    "Gong",
    "Tubular bells",
    "Mallet percussion",
    # 180-189
    "Marimba, xylophone",
    "Glockenspiel",
    "Vibraphone",
    "Steelpan",
    "Orchestra",
    "Brass instrument",
    "French horn",
    "Trumpet",
    "Trombone",
    # 190-199
    "Bowed string instrument",
    "String section",
    "Violin, fiddle",
    "Pizzicato",
    "Cello",
    "Double bass",
    "Wind instrument, woodwind instrument",
    "Flute",
    "Saxophone",
    "Clarinet",
    # 200-209
    "Harp",
    "Bell",
    "Church bell",
    "Jingle bell",
    "Bicycle bell",
    "Tuning fork",
    "Chime",
    "Wind chime",
    "Change ringing (campanology)",
    "Harmonica",
    # 210-219
    "Accordion",
    "Bagpipes",
    "Didgeridoo",
    "Shofar",
    "Theremin",
    "Singing bowl",
    "Scratching (performance technique)",
    "Pop music",
    "Hip hop music",
    "Beatboxing",
    # 220-229
    "Rock music",
    "Heavy metal",
    "Punk rock",
    "Grunge",
    "Progressive rock",
    "Rock and roll",
    "Psychedelic rock",
    "Rhythm and blues",
    "Soul music",
    "Reggae",
    # 230-239
    "Country",
    "Swing music",
    "Bluegrass",
    "Funk",
    "Folk music",
    "Middle Eastern music",
    "Jazz",
    "Disco",
    "Classical music",
    "Opera",
    # 240-249
    "Electronic music",
    "House music",
    "Techno",
    "Dubstep",
    "Drum and bass",
    "Electronica",
    "Electronic dance music",
    "Ambient music",
    "Trance music",
    "Music of Latin America",
    # 250-259
    "Salsa music",
    "Flamenco",
    "Blues",
    "Music for children",
    "New-age music",
    "Vocal music",
    "A capella",
    "Music of Africa",
    "Afrobeat",
    "Christian music",
    # 260-269
    "Gospel music",
    "Music of Asia",
    "Carnatic music",
    "Music of Bollywood",
    "Ska",
    "Traditional music",
    "Independent music",
    "Song",
    "Background music",
    "Theme music",
    # 270-279
    "Jingle (music)",
    "Soundtrack music",
    "Lullaby",
    "Video game music",
    "Christmas music",
    "Dance music",
    "Wedding music",
    "Funny music",
    "Happy music",
    "Sad music",
    # 280-289
    "Tender music",
    "Exciting music",
    "Angry music",
    "Scary music",
    "Wind",
    "Rustling leaves",
    "Wind noise (microphone)",
    "Thunderstorm",
    "Thunder",
    "Water",
    # 290-299
    "Rain",
    "Raindrop",
    "Rain on surface",
    "Stream",
    "Waterfall",
    "Ocean",
    "Waves, surf",
    "Steam",
    "Gurgling",
    "Fire",
    # 300-309
    "Crackle",
    "Vehicle",
    "Boat, Water vehicle",
    "Sailboat, sailing ship",
    "Rowboat, canoe, kayak",
    "Motorboat, speedboat",
    "Ship",
    "Motor vehicle (road)",
    "Car",
    "Vehicle horn, car horn, honking",
    # 310-319
    "Toot",
    "Car alarm",
    "Power windows, electric windows",
    "Skidding",
    "Tire squeal",
    "Car passing by",
    "Race car, auto racing",
    "Truck",
    "Air brake",
    "Air horn, truck horn",
    # 320-329
    "Reversing beeps",
    "Ice cream truck, ice cream van",
    "Bus",
    "Emergency vehicle",
    "Police car (siren)",
    "Ambulance (siren)",
    "Fire engine, fire truck (siren)",
    "Motorcycle",
    "Traffic noise, roadway noise",
    "Rail transport",
    # 330-339
    "Train",
    "Train whistle",
    "Train horn",
    "Railroad car, train wagon",
    "Train wheels squealing",
    "Subway, metro, underground",
    "Aircraft",
    "Aircraft engine",
    "Jet engine",
    "Propeller, airscrew",
    # 340-349
    "Helicopter",
    "Fixed-wing aircraft, airplane",
    "Bicycle",
    "Skateboard",
    "Engine",
    "Light engine (high frequency)",
    "Dental drill, dentist's drill",
    "Lawn mower",
    "Chainsaw",
    "Medium engine (mid frequency)",
    # 350-359
    "Heavy engine (low frequency)",
    "Engine knocking",
    "Engine starting",
    "Idling",
    "Accelerating, revving, vroom",
    "Door",
    "Doorbell",
    "Ding-dong",
    "Sliding door",
    "Slam",
    # 360-369
    "Knock",
    "Tap",
    "Squeak",
    "Cupboard open or close",
    "Drawer open or close",
    "Dishes, pots, and pans",
    "Cutlery, silverware",
    "Chopping (food)",
    "Frying (food)",
    "Microwave oven",
    # 370-379
    "Blender",
    "Water tap, faucet",
    "Sink (filling or washing)",
    "Bathtub (filling or washing)",
    "Hair dryer",
    "Toilet flush",
    "Toothbrush",
    "Electric toothbrush",
    "Vacuum cleaner",
    "Zipper (clothing)",
    # 380-389
    "Keys jangling",
    "Coin (dropping)",
    "Scissors",
    "Electric shaver, electric razor",
    "Shuffling cards",
    "Typing",
    "Typewriter",
    "Computer keyboard",
    "Writing",
    "Alarm",
    # 390-399
    "Telephone",
    "Telephone bell ringing",
    "Ringtone",
    "Telephone dialing, DTMF",
    "Dial tone",
    "Busy signal",
    "Alarm clock",
    "Siren",
    "Civil defense siren",
    "Buzzer",
    # 400-409
    "Smoke detector, smoke alarm",
    "Fire alarm",
    "Foghorn",
    "Whistle",
    "Steam whistle",
    "Mechanisms",
    "Ratchet, pawl",
    "Clock",
    "Tick",
    "Tick-tock",
    # 410-419
    "Gears",
    "Pulleys",
    "Sewing machine",
    "Mechanical fan",
    "Air conditioning",
    "Cash register",
    "Printer",
    "Camera",
    "Single-lens reflex camera",
    "Tools",
    # 420-429
    "Hammer",
    "Jackhammer",
    "Sawing",
    "Filing (rasp)",
    "Sanding",
    "Power tool",
    "Drill",
    "Explosion",
    "Gunshot, gunfire",
    "Machine gun",
    # 430-439
    "Fusillade",
    "Artillery fire",
    "Cap gun",
    "Fireworks",
    "Firecracker",
    "Burst, pop",
    "Eruption",
    "Boom",
    "Wood",
    "Chop",
    # 440-449
    "Splinter",
    "Crack",
    "Glass",
    "Chink, clink",
    "Shatter",
    "Liquid",
    "Splash, splatter",
    "Slosh",
    "Squish",
    "Drip",
    # 450-459
    "Pour",
    "Trickle, dribble",
    "Gush",
    "Fill (with liquid)",
    "Spray",
    "Pump (liquid)",
    "Stir",
    "Boiling",
    "Sonar",
    "Arrow",
    # 460-469
    "Whoosh, swoosh, swish",
    "Thump, thud",
    "Thunk",
    "Electronic tuner",
    "Effects unit",
    "Chorus effect",
    "Basketball bounce",
    "Bang",
    "Slap, smack",
    "Whack, thwack",
    # 470-479
    "Smash, crash",
    "Breaking",
    "Bouncing",
    "Whip",
    "Flap",
    "Scratch",
    "Scrape",
    "Rub",
    "Roll",
    "Crushing",
    # 480-489
    "Crumpling, crinkling",
    "Tearing",
    "Beep, bleep",
    "Ping",
    "Ding",
    "Clang",
    "Squeal",
    "Creak",
    "Rustle",
    "Whir",
    # 490-499
    "Clatter",
    "Sizzle",
    "Clicking",
    "Clickety-clack",
    "Rumble",
    "Plop",
    "Jingle, tinkle",
    "Hum",
    "Zing",
    "Boing",
    # 500-509
    "Crunch",
    "Silence",
    "Sine wave",
    "Harmonic",
    "Chirp tone",
    "Sound effect",
    "Pulse",
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    # 510-519
    "Outside, urban or manmade",
    "Outside, rural or natural",
    "Reverberation",
    "Echo",
    "Noise",
    "Environmental noise",
    "Static",
    "Mains hum",
    "Distortion",
    "Sidetone",
    # 520-526
    "Cacophony",
    "White noise",
    "Pink noise",
    "Throbbing",
    "Vibration",
    "Television",
    "Radio",
    "Field recording",
]

assert len(_AUDIOSET_LABELS) == 527, f"Expected 527 labels, got {len(_AUDIOSET_LABELS)}"


class AudioSetLabels:
    """Wrapper around the 527 AudioSet labels with lookup methods."""

    def __init__(self) -> None:
        self._labels = _AUDIOSET_LABELS
        self._index_map = {label: i for i, label in enumerate(self._labels)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> str:
        return self._labels[idx]

    def __contains__(self, label: str) -> bool:
        return label in self._index_map

    def index(self, label: str) -> int:
        if label not in self._index_map:
            msg = f"Label {label!r} not found in AudioSet labels"
            raise ValueError(msg)
        return self._index_map[label]


# Label groups: each maps to a single HA binary_sensor entity.
# Multiple AudioSet labels collapse into one group, taking the highest confidence.
#
# Groups are organized into categories:
#   - Safety & security: smoke_alarm, glass_break, siren, gunshot_explosion, screaming
#   - People & pets: dog_bark, cat_meow, crying, speech, cough_sneeze, footsteps
#   - Doors & entry: doorbell, knock, door, cabinet
#   - Environment: rain_storm, music, vehicle, car_horn
#   - Household: vacuum_cleaner, water_running, kitchen_appliance, power_tools, alarm_beep
#   - Equipment monitoring: hvac_mechanical, mechanical_anomaly, water_leak, electrical_anomaly
LABEL_GROUPS: dict[str, list[str]] = {
    # --- Safety & security ---
    "smoke_alarm": [
        "Smoke detector, smoke alarm",
        "Fire alarm",
    ],
    "glass_break": [
        "Shatter",
        "Smash, crash",
        "Breaking",
    ],
    "siren": [
        "Siren",
        "Civil defense siren",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
    ],
    "gunshot_explosion": [
        "Gunshot, gunfire",
        "Explosion",
        "Boom",
        "Fireworks",
        "Firecracker",
    ],
    "screaming": [
        "Screaming",
        "Shout",
        "Yell",
        "Battle cry",
        "Children shouting",
    ],
    # --- People & pets ---
    "dog_bark": [
        "Dog",
        "Bark",
        "Bow-wow",
        "Howl",
        "Growling",
        "Whimper (dog)",
        "Yip",
    ],
    "cat_meow": [
        "Cat",
        "Purr",
        "Meow",
        "Hiss",
        "Caterwaul",
    ],
    "crying": [
        "Baby cry, infant cry",
        "Crying, sobbing",
        "Wail, moan",
    ],
    "speech": [
        "Speech",
        "Conversation",
        "Child speech, kid speaking",
    ],
    "cough_sneeze": [
        "Cough",
        "Sneeze",
        "Throat clearing",
        "Wheeze",
    ],
    "footsteps": [
        "Walk, footsteps",
        "Run",
        "Shuffle",
    ],
    # --- Doors & entry ---
    "doorbell": [
        "Doorbell",
        "Ding-dong",
    ],
    "knock": [
        "Knock",
        "Tap",
    ],
    "door": [
        "Door",
        "Sliding door",
        "Slam",
    ],
    "cabinet": [
        "Cupboard open or close",
        "Drawer open or close",
    ],
    # --- Environment ---
    "rain_storm": [
        "Rain",
        "Raindrop",
        "Rain on surface",
        "Thunderstorm",
        "Thunder",
        "Wind",
    ],
    "music": [
        "Music",
        "Pop music",
        "Rock music",
        "Classical music",
        "Electronic music",
        "Jazz",
        "Hip hop music",
    ],
    "vehicle": [
        "Vehicle",
        "Motor vehicle (road)",
        "Car",
        "Car passing by",
        "Truck",
        "Bus",
        "Motorcycle",
        "Traffic noise, roadway noise",
    ],
    "car_horn": [
        "Vehicle horn, car horn, honking",
        "Toot",
        "Air horn, truck horn",
        "Car alarm",
    ],
    "aircraft": [
        "Aircraft",
        "Aircraft engine",
        "Jet engine",
        "Propeller, airscrew",
        "Helicopter",
        "Fixed-wing aircraft, airplane",
    ],
    # --- Household ---
    "vacuum_cleaner": [
        "Vacuum cleaner",
    ],
    "water_running": [
        "Water tap, faucet",
        "Sink (filling or washing)",
        "Bathtub (filling or washing)",
        "Toilet flush",
        "Splash, splatter",
        "Pour",
    ],
    "kitchen_appliance": [
        "Microwave oven",
        "Blender",
        "Dishes, pots, and pans",
        "Boiling",
        "Frying (food)",
        "Sizzle",
        "Chopping (food)",
        "Cutlery, silverware",
    ],
    "power_tools": [
        "Power tool",
        "Drill",
        "Chainsaw",
        "Lawn mower",
        "Jackhammer",
        "Sawing",
        "Hammer",
        "Sanding",
    ],
    "alarm_beep": [
        "Alarm",
        "Alarm clock",
        "Buzzer",
        "Beep, bleep",
        "Reversing beeps",
    ],
    # --- Equipment monitoring (failure detection) ---
    "hvac_mechanical": [
        "Mechanical fan",
        "Air conditioning",
        "Engine",
        "Idling",
    ],
    "mechanical_anomaly": [
        "Engine knocking",
        "Squeal",
        "Creak",
        "Rattle",
        "Clicking",
        "Clatter",
        "Rumble",
        "Vibration",
        "Throbbing",
    ],
    "water_leak": [
        "Drip",
        "Trickle, dribble",
        "Gush",
    ],
    "electrical_anomaly": [
        "Buzz",
        "Mains hum",
        "Distortion",
    ],
    # --- Media (TV/movie audio — used as confuser alternative) ---
    "media": [
        "Television",
        "Radio",
    ],
}

# Reverse lookup: label -> group name
_LABEL_TO_GROUP: dict[str, str] = {}
for _group_name, _labels in LABEL_GROUPS.items():
    for _label in _labels:
        _LABEL_TO_GROUP[_label] = _group_name


def get_group_for_label(label: str) -> str | None:
    """Return the group name for a given AudioSet label, or None if ungrouped."""
    return _LABEL_TO_GROUP.get(label)


def get_top_group_match(
    predictions: list[tuple[str, float]],
    threshold: float = 0.15,
    *,
    all_groups: bool = False,
) -> tuple[str, float, str] | list[tuple[str, float, str]] | None:
    """Find the best matching group(s) from a list of (label, score) predictions.

    Args:
        predictions: List of (label, score) tuples from the classifier.
        threshold: Minimum confidence to consider a match.
        all_groups: If True, return all matching groups sorted by confidence.

    Returns:
        Single (group, confidence, raw_label) tuple, or list if all_groups=True,
        or None if no match above threshold.
    """
    # Collect best score per group
    group_best: dict[str, tuple[float, str]] = {}
    for label, score in predictions:
        group = get_group_for_label(label)
        if group is None:
            continue
        if score < threshold:
            continue
        if group not in group_best or score > group_best[group][0]:
            group_best[group] = (score, label)

    if not group_best:
        return [] if all_groups else None

    results = [
        (group, score, raw_label) for group, (score, raw_label) in group_best.items()
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    if all_groups:
        return results
    return results[0]
