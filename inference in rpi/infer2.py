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

        self.temporal_pe = nn.Embedding(time_in, dmodel)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(time_in, 2)

    def forward(self, x):

        b, t, c = x.shape

        x = self.input_embedding(x)

        pe = self.temporal_pe(self.temporal_pos).expand(
            b, t, -1
        )

        x = x + pe

        x = self.temporal_encoder(x)

        x = self.decoder(x)

        x = self.flatten(x)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "./models/TFEQ_state_dict.pth"

BSON_PATH = "./bson_file/smallAccelerogram.bson"

WINDOW_SIZE = 200

STEP_SIZE = 100


# =====================================================
# LOAD MODEL
# =====================================================

model = TFEQ(channel=3, time_in=200)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded\n")


# =====================================================
# READ ONLY FIRST EVENT
# =====================================================

events = {}

with open(BSON_PATH, "rb") as f:

    for doc in decode_file_iter(f):

        event_id = doc["_id"].rsplit(".", 1)[0]

        component = doc["_id"].split(".")[-1]

        if event_id not in events:
            events[event_id] = {}

        events[event_id][component] = np.array(
            doc["accelerogram"],
            dtype=np.float32
        )

        # Wait until all channels arrive

        if not all(
            x in events[event_id]
            for x in ["UD", "NS", "EW"]
        ):
            continue

        print("=" * 60)
        print("Event:", event_id)

        comp = events[event_id]

        min_length = min(
            len(comp["UD"]),
            len(comp["NS"]),
            len(comp["EW"])
        )

        ud = comp["UD"][:min_length]
        ns = comp["NS"][:min_length]
        ew = comp["EW"][:min_length]

        windows = []

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

        windows = np.array(windows)

        windows = np.transpose(
            windows,
            (0, 2, 1)
        )

        print(
            f"\nNumber of windows: {len(windows)}\n"
        )

        predictions = []

        window_times = []

        with torch.no_grad():

            for i, window in enumerate(windows):

                x = torch.tensor(
                    window,
                    dtype=torch.float32
                ).unsqueeze(0)

                start_time = time.time()

                output = model(x)

                end_time = time.time()

                probs = torch.exp(output)

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

                inference_time = (
                    end_time - start_time
                )

                predictions.append(pred)

                window_times.append(
                    inference_time
                )

                print(
                    f"Window {i + 1:02d} | "
                    f"Prediction = {pred} | "
                    f"Time = {inference_time:.6f} sec"
                )

        print("\nSummary")

        print(
            "Predictions:",
            predictions
        )

        print(
            f"Average time: "
            f"{np.mean(window_times):.6f} sec"
        )

        print(
            f"Maximum time: "
            f"{np.max(window_times):.6f} sec"
        )

        print(
            f"Minimum time: "
            f"{np.min(window_times):.6f} sec"
        )

        # Stop after first event

        break


print("\nFinished.")