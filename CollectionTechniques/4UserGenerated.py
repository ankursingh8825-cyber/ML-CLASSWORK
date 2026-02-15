import csv
import random
from datetime import datetime

output_file = "../csv_files/service_logs.csv"

services = ["auth", "payment", "search", "profile", "notifications"]

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["timestamp", "service_name", "status", "response_time_ms"]
    )
    writer.writeheader()

    for _ in range(100):  # 100 log lines
        service = random.choice(services)
        response_time = random.randint(50, 700)   # ms
        is_error = random.random() < 0.1          # 10% error chance

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "service_name": service,
            "status": "ERROR" if is_error else "OK",
            "response_time_ms": response_time,
        }

        writer.writerow(row)

print(f"Generated service logs in {output_file}")
