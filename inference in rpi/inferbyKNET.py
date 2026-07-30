import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bson import decode_file_iter


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
        super().__init__()

        self.input_embedding = nn.Linear(channel, dmodel)

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

        self.decoder = nn.Linear(dmodel, 1)

        self.temporal_pos = torch.arange(0, time_in)

        self.temporal_pe = nn.Embedding(
            time_in,
            dmodel
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(time_in, 2)

    def forward(self, x):

        b, t, c = x.shape

        x = self.input_embedding(x)

        pe = self.temporal_pe(
            self.temporal_pos
        ).expand(b, t, -1)

        x = x + pe

        x = self.temporal_encoder(x)

        x = self.decoder(x)

        x = self.flatten(x)

        x = self.fc(x)

        return F.log_softmax(
            x,
            dim=1
        )


# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "./models/TFEQ_state_dict.pth"

BSON_PATH = "./bson_file/small_KNET1.bson"

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

print("Model loaded.\n")


# =====================================================
# READ EVENTS
# =====================================================

events = {}

with open(BSON_PATH, "rb") as f:

    for doc in decode_file_iter(f):

        filename = doc["FileName"]

        event_id, component = filename.split(".")

        # Keep only sensor-1 channels
        if component not in [
            "EW1",
            "NS1",
            "UD1"
        ]:
            continue

        if event_id not in events:
            events[event_id] = {}

        # Store both signal and PGA
        events[event_id][component] = {

            "signal": np.array(
                doc["New_Acceleration"],
                dtype=np.float32
            ),

            "pga": float(
                doc["HeaderInfo"]["Max. Acc. (gal)"]
            )
        }

        # Wait until all three channels arrive
        if not all(
            x in events[event_id]
            for x in [
                "EW1",
                "NS1",
                "UD1"
            ]
        ):
            continue

        print("=" * 70)

        comp = events[event_id]

        event_pga = max(
            comp["EW1"]["pga"],
            comp["NS1"]["pga"],
            comp["UD1"]["pga"]
        )

        print("Event :", event_id)
        print("PGA   :", event_pga, "gal")

        # -------------------------------------------------
        # GET SIGNALS
        # -------------------------------------------------

        min_length = min(

            len(
                comp["EW1"]["signal"]
            ),

            len(
                comp["NS1"]["signal"]
            ),

            len(
                comp["UD1"]["signal"]
            )
        )

        ew = comp["EW1"]["signal"][:min_length]

        ns = comp["NS1"]["signal"][:min_length]

        ud = comp["UD1"]["signal"][:min_length]

        # -------------------------------------------------
        # CREATE WINDOWS
        # -------------------------------------------------

        windows = []

        for start in range(

            0,

            min_length
            - WINDOW_SIZE + 1,

            STEP_SIZE
        ):

            end = start + WINDOW_SIZE

            window = np.stack(

                [

                    ew[start:end],

                    ns[start:end],

                    ud[start:end]

                ],

                axis=0
            )

            windows.append(
                window
            )

        windows = np.array(
            windows,
            dtype=np.float32
        )

        # (N, 3, 200)
        # ->
        # (N, 200, 3)

        windows = np.transpose(

            windows,

            (0, 2, 1)
        )

        x = torch.tensor(
            windows
        )

        # -------------------------------------------------
        # INFERENCE
        # -------------------------------------------------

        start_time = time.time()

        with torch.no_grad():

            output = model(
                x
            )

            probs = torch.exp(
                output
            )

            pred = torch.argmax(

                probs,

                dim=1
            )

        end_time = time.time()

        total_time = (

            end_time
            - start_time
        )

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------

        earthquake_windows = int(
            (pred == 1).sum()
        )

        non_earthquake_windows = int(
            (pred == 0).sum()
        )

        print(
            "Windows:",
            len(windows)
        )

        print(
            "Earthquake windows:",
            earthquake_windows
        )

        print(
            "Non-earthquake windows:",
            non_earthquake_windows
        )

        print(
            "Predictions:"
        )

        print(
            pred.tolist()
        )

        print(
            f"Total inference time: "
            f"{total_time:.6f} sec"
        )

        print(
            f"Time per window: "
            f"{total_time / len(windows):.6f} sec"
        )

        del events[event_id]

print("\nFinished.")