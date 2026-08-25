import pandas as pd
import os

def create_initial_dataset():
    # Example seed data: Roller coaster text often has erratic punctuation or fragmented structure 
    # due to wind and motion, compared to calm dictation.
    data = {
        "text": [
            "The quick brown fox jumps over the lazy dog while sitting at my desk.",
            "Ahhhhh omg we are dropping down the hill so fast I can't breathe ahhh!",
            "Please schedule a meeting for tomorrow at 10 AM to discuss the quarterly report.",
            "Whoa holy crap look at that loop—wait stop typing my phone is shaking so much AAAH!",
            "I need to pick up groceries on my way home from work today."
        ],
        "is_roller_coaster": [0, 1, 0, 1, 0] # 1 = Roller coaster, 0 = Normal
    }
    
    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/samples.csv", index=False)
    print("Initial dataset created successfully at data/raw/samples.csv!")

if __name__ == "__main__":
    create_initial_dataset()
