import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_fake_transactions(n=1000, seed=42):
    np.random.seed(seed)
    
    data = {
        "user_id": np.random.randint(1000, 1100, size=n),
        "amount": np.random.exponential(scale=100, size=n),
        "location": np.random.choice(['US', 'UK', 'IN', 'DE', 'FR'], size=n),
        "timestamp": [datetime.now() - timedelta(minutes=np.random.randint(0, 100000)) for _ in range(n)]
    }

    df = pd.DataFrame(data)
    
    anomalies = df.sample(frac=0.02)
    df.loc[anomalies.index, "amount"] *= 10

    df.to_csv("transactions.csv", index=False)

if __name__ == "__main__":
    generate_fake_transactions()