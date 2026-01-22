import csv

# Create/Open CSV file
with open("data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Header
    writer.writerow(["Name", "Age", "City"])

    # Take 5 inputs
    for i in range(5):
        print(f"\nEnter details for person {i+1}:")
        name = input("Enter Name: ")
        age = input("Enter Age: ")
        city = input("Enter City: ")

        writer.writerow([name, age, city])

print("\nCSV file created successfully with 5 entries!")
