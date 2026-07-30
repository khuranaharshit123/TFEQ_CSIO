import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# MODEL
# =====================================================

class TFEQ(nn.Module):

    def __init__(
        self,
        channel=3,
        dmodel=64,
        nhead=8,
        dim_feedforward=32,
        num_layers=1,
        time_in=200
    ):
        super(TFEQ, self).__init__()

        self.input_embedding = nn.Linear(
            channel,
            dmodel
        )

        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            dmodel,
            nhead,
            dim_feedforward,
            batch_first=True
        )

        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer,
            num_layers
        )

        self.decoder = nn.Linear(
            dmodel,
            1
        )

        self.temporal_pos = torch.arange(
            0,
            time_in
        )

        self.temporal_pe = nn.Embedding(
            time_in,
            dmodel
        )

        self.fla = nn.Flatten()

        self.dp = nn.Dropout(
            p=0.5
        )

        self.fc = nn.Linear(
            time_in,
            2
        )

    def forward(self, x):

        b, t, c = x.shape

        x = self.input_embedding(x)

        t_pe = self.temporal_pe(
            self.temporal_pos
        ).expand(
            b,
            t,
            -1
        )

        x = x + t_pe

        x = self.temporal_encoder(x)

        x = self.decoder(x)

        x = self.fla(x)

        x = self.fc(x)

        return F.log_softmax(
            x,
            dim=1
        )


# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "./models/TFEQ_state_dict.pth"

JSON_PATH = "./bson_file/knet_1530.dataaccelerogram.json"

WINDOW_SIZE = 200

STEP_SIZE = 100


# =====================================================
# LOAD MODEL
# =====================================================

model = TFEQ(
    channel=3,
    time_in=200
)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    state_dict
)

model.eval()

print("Model loaded\n")


# =====================================================
# LOAD JSON
# =====================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# =====================================================
# PROCESS 10 EVENTS
# =====================================================

for event_number, doc in enumerate(data[:10], start=1):

    event_id = doc["_id"]

    print("\n" + "=" * 60)

    print(
        f"Event {event_number}:",
        event_id
    )

    comp = {}

    # Extract signals

    for signal in doc["signals"]:

        component = signal["_id"].split(".")[-1]

        comp[component] = np.array(

            signal["accelerogram"],

            dtype=np.float32

        )

    # Skip incomplete events

    if not all(

        x in comp

        for x in ["UD", "NS", "EW"]

    ):

        print(
            "Missing component"
        )

        continue

    # Match lengths

    min_length = min(

        len(comp["UD"]),

        len(comp["NS"]),

        len(comp["EW"])

    )

    ud = comp["UD"][:min_length]

    ns = comp["NS"][:min_length]

    ew = comp["EW"][:min_length]

    windows = []

    # Create windows

    for start in range(

        0,

        min_length - WINDOW_SIZE + 1,

        STEP_SIZE

    ):

        end = start + WINDOW_SIZE

        window = np.stack([

            ew[start:end],

            ns[start:end],

            ud[start:end]

        ])

        windows.append(window)

    windows = np.array(

        windows

    )

    windows = np.transpose(

        windows,

        (0, 2, 1)

    )

    x = torch.tensor(

        windows,

        dtype=torch.float32

    )

    start_time = time.time()

    with torch.no_grad():

        output = model(x)

        probs = torch.exp(

            output

        )

        pred = torch.argmax(

            probs,

            dim=1

        )

    end_time = time.time()

    total_time = (

        end_time -

        start_time

    )

    time_per_window = (

        total_time /

        len(windows)

    )

    print(

        "Number of windows:",

        len(windows)

    )

    print(

        "Prediction:",

        pred.tolist()

    )

    print(

        f"Total inference time: "

        f"{total_time:.6f} sec"

    )

    print(

        f"Time per window: "

        f"{time_per_window:.6f} sec"

    )

print("\nFinished.")