import csv
with open("csv_files/data.csv", mode="w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    for i in range(5):
        print(f"Enter details for person {i+1}:")
        name = input("Enter Name: ")
        age = input("Enter Age: ")
        city = input("Enter City: ")
        writer.writerow([name, age, city])

print("CSV file created successfully")

