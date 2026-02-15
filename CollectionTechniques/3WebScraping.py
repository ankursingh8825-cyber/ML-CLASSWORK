import requests
import csv

url = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(url)

words = response.text.split()

rows = [{"word": w} for w in words[:100]]  # first 100 words

with open("../csv_files/words.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["word"])
    writer.writeheader()
    writer.writerows(rows)

print("Saved words!")
