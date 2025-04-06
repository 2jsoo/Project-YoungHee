from import_library import *
import shutil

def normalize_pose(lm_array):
    # Center around hips
    center = (lm_array[23][:2] + lm_array[24][:2]) / 2
    centered = lm_array[:, :2] - center
    # Scale by shoulder-to-ankle length
    scale = np.linalg.norm(centered[11] - centered[27])
    scaled = centered / scale if scale > 0 else centered
    return scaled.flatten()

class DataProcessing:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    def download_kth_dataset(self):
        """Download KTH dataset for specified actions"""
        # Create base directories
        data_folder = "data/KTH"
        os.makedirs(data_folder, exist_ok=True)
        
        # Define action actegories for game
        actions = ['boxing', 'handclapping', 'handwaving', 'walking']
        
        # KTH dataset URL 
        base_url = "https://www.csc.kth.se/cvap/actions"

        print("Downloading KTH dataset (selected actions only)...")
        
        # Download each action ZIP file
        for action in actions:
            action_dir = os.path.join(data_folder, action)
            os.makedirs(action_dir, exist_ok=True)
            
            # URL for action
            zip_url = f"{base_url}/{action}.zip"
            zip_path = os.path.join(data_folder, f"{action}.zip")
            
            try:
                # Download ZIP file
                print(f"Downloading {action}.zip...")
                response = requests.get(zip_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as file, tqdm(
                    desc=zip_url,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))
                
                # Extract ZIP file
                print(f"Extracting {action}.zip...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.endswith('.avi'):
                            # Save .avi file to directories
                            source = zip_ref.open(file)
                            target = open(os.path.join(action_dir, os.path.basename(file)), "wb")
                            with source, target:
                                shutil.copyfileobj(source, target)
                
                # Remove ZIP file
                os.remove(zip_path)
                print(f"Downloaded and extracted {action} videos.")
                
            except Exception as e:
                print(f"Error downloading/extracting {action}: {e}")
        
        # Check downloaded files
        total_files = 0
        for action in actions:
            action_dir = os.path.join(data_folder, action)
            files = [f for f in os.listdir(action_dir) if f.endswith('.avi')]
            total_files += len(files)
            print(f"Action {action}: {len(files)} video files")
        
        print(f"Total downloaded: {total_files} video files")
    
    def extract_pose_features(self, video_path, num_frames=30):
        """Extract pose features from video using MediaPipe"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        # Select frames (=30) evenly
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        features = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            # Extract 33 landmarks with x, y, z, visibility
            landmarks = results.pose_landmarks.landmark
            lm_array = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks])
            
            # Normalization
            normalized_features = normalize_pose(lm_array)
            features.append(normalized_features)
        
        cap.release()
        
        # Zero padding (if less than num_frames)
        while len(features) < num_frames:
            features.append(np.zeros_like(features[0]))
            
        return np.array(features)
    
    def process_kth_for_younghee(self, num_frames=30, download_if_missing=True):
        """Process KTH dataset"""
        print("Processing KTH dataset...")
        
        # Data path
        data_folder = "data/KTH"
        save_path = "data/processed/younghee"
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        
        # Define action actegories
        categories = ['boxing', 'handclapping', 'handwaving', 'walking']
        class_idx_dict = {category: i for i, category in enumerate(categories)}
        
        # Check data exists
        if not os.path.exists(f'{save_path}/{categories[0]}'):
            print('Download')
            self.download_kth_dataset()
        
        # Process KTH
        features = []
        labels = []
        for category in tqdm(categories, desc="Processing categories"):
            category_folder = os.path.join(data_folder, category)
            
            files = [f for f in os.listdir(category_folder) if f.endswith('.avi')]
            if len(files) == 0:
                print(f"No {category} files")
                continue
                
            class_idx = class_idx_dict[category]
            
            # Extract features from files
            for file in tqdm(files, desc=f"Processing {category}"):
                try:
                    pose_features = self.extract_pose_features(os.path.join(category_folder, file), num_frames)
                    
                    if pose_features is not None:
                        features.append(pose_features)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
        
        if len(features) == 0:
            print("No features extracted")
        
        # Train Test split
        X, y = np.array(features), np.array(labels)
        
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev)
        
        normalized_shape = X_train[0].shape
        
        dataset = {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'X_test': X_test, 
            'y_test': y_test,
            'class_names': categories,
            'normalization': {
                'method': 'custom_hip_center_shoulder_ankle_scale',
                'feature_dim': normalized_shape[-1]
            }
        }
        
        # Save dataset
        with open(os.path.join(save_path, "dataset.pkl"), 'wb') as f:
            pickle.dump(dataset, f)
        
        unique_labels, counts = np.unique(y, return_counts=True)
        class_counts = {categories[lbl]: count for lbl, count in zip(unique_labels, counts)}
        print(f"Total samples: {len(X)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        
        print(f"Save Path: {os.path.join(save_path, 'dataset.pkl')}")
        print(f"Feature dimension: {normalized_shape[-1]}")
        
        return dataset