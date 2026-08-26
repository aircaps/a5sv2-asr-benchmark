TARGET_WORDS = 32_928

MEGA = {
    "id": "zhifeixie/Voices-in-the-Wild-2M",
    "revision": "a8a35d3319737190d6fd3d39157b258eaab35980",
    "seed": 20260823,
    "shuffle_buffer": 512,
    "examples_per_condition": 250,
    "conditions": ["noise", "far_field", "far_field_noise", "recording_noise", "obstructed_noise"],
    "shards": {
        "noise": (131, [3, 17, 31, 47, 63, 79, 101, 127]),
        "far_field": (57, [0, 7, 14, 21, 28, 35, 42, 56]),
        "far_field_noise": (131, [3, 17, 31, 47, 63, 79, 101, 127]),
        "recording_noise": (3, [0, 1, 2]),
        "obstructed_noise": (3, [0, 1, 2]),
    },
}

AMI = {
    "revision": "manual-annotations-1.6.2",
    "meetings": ["ES2004d", "ES2014a", "IS1009a", "IS1009b", "TS3003b", "TS3003c", "TS3007b"],
    "channel": "Array1-01",
}

DIPCO = {
    "revision": "e2b29d3d0d88692c744feb15e290f7316b68014e",
    "repo": "huckiyang/DiPCo",
    "eval_mirror": "vidalfernando/dipco_eval",
    "eval_mirror_revision": "9eaeeb8264fd655401ae77531784e4aad7fe6611",
    "mirror_shards": {"S03": 5, "S06": 9},
    "meetings": {"eval": ["S01", "S03", "S06", "S07", "S08"], "dev": ["S02"]},
    "device": "U01",
    "channel": "CH1",
    "gain_db": 5.1,
}

NOTSOFAR = {
    "revision": "ba8fd0f034ce185fe4d24f47e53b4b8194795f07",
    "version": "240629.1_eval_small_with_GT",
    "meetings": [
        "MTG_32102",
        "MTG_32007",
        "MTG_32068",
        "MTG_32022",
        "MTG_32048",
        "MTG_32069",
        "MTG_32080",
        "MTG_32107",
        "MTG_32072",
        "MTG_32088",
        "MTG_32178",
        "MTG_32108",
        "MTG_32105",
        "MTG_32082",
        "MTG_32003",
        "MTG_32071",
        "MTG_32087",
        "MTG_32000",
        "MTG_32106",
        "MTG_32026",
        "MTG_32322",
        "MTG_32004",
    ],
    "audio": "sc_meetup_0/ch0.wav",
    "selection_seed": "notsofar-production-v1",
}

DATASET_ID = MEGA["id"]
DATASET_REVISION = MEGA["revision"]
CONDITIONS = MEGA["conditions"]
REPORT_CONDITIONS = sorted(CONDITIONS)
