import json
from tabulate import tabulate

def load_logs(file="diagnosis_logs.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def display_all():
    logs = load_logs()
    if not logs:
        print("No diagnosis history found.")
        return

    table = [
        [log["timestamp"], log["username"], log["label"], f'{log["confidence"]:.2f}%', log["description"]]
        for log in logs
    ]
    headers = ["Timestamp", "Username", "Diagnosis", "Confidence", "Description"]
    print(tabulate(table, headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    display_all()
