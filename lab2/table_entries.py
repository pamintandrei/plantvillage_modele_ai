import os
from openpyxl import Workbook

leafTypes = []

workbook = Workbook()
sheet = workbook.active
sheet["A1"] = 'Class'
sheet["B1"] = 'Number of entries'
for root, dirs, files in os.walk("./images/PlantVillage"):
    for index, name in enumerate(dirs):
        class_name = name.split('\\')[-1]
        leafTypes.append(class_name)
        number_of_entries = len(os.listdir(f"./images/PlantVillage/{name}/"));
        # allfiles = os.walk(name)
        print(f"Class name: {class_name} " + str(number_of_entries))
        sheet[f"A{index + 2}"] = class_name
        sheet[f"B{index + 2}"] = number_of_entries

workbook.save(filename="classes.xlsx")
