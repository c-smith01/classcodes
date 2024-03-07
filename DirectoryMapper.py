import os
import folium

def create_map(directory_path):
    # Create map centered at the directory location
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Iterate through files and folders in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Get latitude and longitude based on file path
            lat, lon = hash(file_path) % 180 - 90, hash(file_path) % 360 - 180
            # Create marker for each file
            folium.Marker([lat, lon], tooltip=file, popup=file_path).add_to(m)

    return m

if __name__ == "__main__":
    # Enter the path of the directory you want to visualize
    directory_path = input("Enter the directory path: ")
    if os.path.isdir(directory_path):
        map_filename = "directory_map.html"
        map_obj = create_map(directory_path)
        map_obj.save(map_filename)
        print(f"Map saved as '{map_filename}'")
    else:
        print("Invalid directory path.")
