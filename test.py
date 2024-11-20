import os
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Step 1: Load the CSV file containing image paths and labels
csv_file = r'C:\Users\shruti shreya\monitor_env\Crop_details.csv'
df = pd.read_csv(csv_file)

# Step 2: Define the base directory for original and augmented images
base_dir = r'C:\Users\shruti shreya\monitor_env'  # Main directory
train_dir = os.path.join(base_dir, 'train_data')  # Directory for training data
test_dir = os.path.join(base_dir, 'test_data')    # Directory for testing data

# Step 3: Create necessary subdirectories for each crop
def create_directories(df, base_dir):
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    crops = df['crop'].unique()
    
    for crop in crops:
        os.makedirs(os.path.join(train_dir, crop), exist_ok=True)
        os.makedirs(os.path.join(test_dir, crop), exist_ok=True)

create_directories(df, base_dir)

# Step 4: Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 5: Move images to their corresponding directories
def move_images(df, target_dir):
    for _, row in df.iterrows():
        img_path = row['path'].lstrip('/')  # Strip the leading slash
        img_path = os.path.join(base_dir, img_path)  # Build full path to image
        
        if os.path.exists(img_path):
            crop = row['crop']
            target_path = os.path.join(target_dir, crop, os.path.basename(img_path))
            shutil.copy(img_path, target_path)
        else:
            print(f"Warning: {img_path} does not exist.")

move_images(train_df, train_dir)
move_images(test_df, test_dir)

# Step 6: Use ImageDataGenerator for data augmentation
Data = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Step 7: Create ImageDataGenerators for training and testing sets
Train_Data1 = Data.flow_from_directory(
    train_dir,  # Path to the training data directory
    target_size=(224, 224),  # Resize images to match the model input
    batch_size=32,  # Batch size for training
    class_mode='categorical'  # For multi-class classification
)

Test_Data = Data.flow_from_directory(
    test_dir,  # Path to the testing data directory
    target_size=(224, 224),  # Resize images to match the model input
    batch_size=32,  # Batch size for testing
    class_mode='categorical'  # For multi-class classification
)

# Step 8: Display the classes in the dataset
print(f"Training classes: {Train_Data1.class_indices}")
print(f"Test classes: {Test_Data.class_indices}")
