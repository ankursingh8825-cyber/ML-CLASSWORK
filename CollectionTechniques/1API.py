import requests
import csv

pokemon_names = ["pikachu", "bulbasaur", "charmander", "squirtle"]

rows = []

for name in pokemon_names:
    data = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name}").json()

    rows.append({
        "name": data["name"],
        "height": data["height"],
        "weight": data["weight"],
        "base_experience": data["base_experience"]
    })

with open("../csv_files/pokemon_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved multiple Pokémon data!")
