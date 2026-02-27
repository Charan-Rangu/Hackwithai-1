from ultralytics import YOLO

ppe = YOLO('foodsafety.pt')
kitchen = YOLO('kitchenobj.pt')

print("PPE:")
for k, v in ppe.names.items():
    print(f"{k}: {v}")

print("Kitchen:")
for k, v in kitchen.names.items():
    print(f"{k}: {v}")
