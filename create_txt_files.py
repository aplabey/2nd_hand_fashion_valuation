import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('raw_data/vestiaire.csv')

# Create a directory to store the txt files
output_dir = 'unique_values_txt_files'
os.makedirs(output_dir, exist_ok=True)

# List of columns and corresponding filenames for storing unique values
columns_and_filenames = {
    'brand_name': 'uv_brands.txt',
    'product_material': 'uv_materials.txt',
    'product_gender_target': 'uv_gender.txt',
    'product_color': 'uv_colors.txt',
    'product_condition': 'uv_condition.txt',
    'usually_ships_within': 'uv_shipping_days.txt',
    'product_category': 'uv_category.txt',
    'product_season': 'uv_season.txt',
    'seller_badge': 'uv_seller_badge.txt'
}

# Iterate over the dictionary to extract unique values and save them to txt files
for column, file_name in columns_and_filenames.items():
    unique_values = df[column].unique()  # Get unique values for the column
    file_path = os.path.join(output_dir, file_name)  # Define the file path

    with open(file_path, 'w') as f:
        for value in unique_values:
            f.write(str(value) + '\n')

print("Text files created successfully.")
