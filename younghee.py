from import_library import *
from trainer import ActionClassifier

def normalize_pose(lm_array):
    # Center around hips
    center = (lm_array[23][:2] + lm_array[24][:2]) / 2
    centered = lm_array[:, :2] - center
    # Scale by shoulder-to-ankle length
    scale = np.linalg.norm(centered[11] - centered[27])
    scaled = centered / scale if scale > 0 else centered
    return scaled.flatten()

class YoungHee:
    def __init__(self, model_path=None, difficulty='medium', input_size=66, num_classes=4):
        # Game state
        self.red_light = False
        self.game_over = False
        self.current_required_action = None
        self.buffer_size = 30  # 30 frames 
        self.set_difficulty(difficulty)
        
        # Player state
        self.player_score = 0
        self.player_sequence = []
        self.penalized = False
        self.completed_action = False
        self.prev_landmarks = None
        
        # Action detection
        self.current_action_idx = -1
        self.current_action_name = None
        self.current_confidence = 0.0
        self.last_completed_action = None  # Store the last completed action
        
        # Feedback
        self.action_result = None
        self.feedback_timestamp = 0
        self.feedback_duration = 1.5

        self.prev_action_idx = None
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load model
        self.model = None
        self.class_names = ['boxing', 'handclapping', 'handwaving', 'walking']
        
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                hidden_size = checkpoint.get('hidden_size', 128)
                num_layers = checkpoint.get('num_layers', 2)
                dropout = checkpoint.get('dropout', 0.3)
                
                self.model = ActionClassifier(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    dropout=dropout
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'class_names' in checkpoint:
                    self.class_names = checkpoint['class_names']
                
                self.model.eval()
                print(f"Model loaded. Classes: {self.class_names}")
            except Exception as e:
                print(f"Failed to load model: {e}")
    
    def set_difficulty(self, difficulty):
        if difficulty == 'easy':
            self.light_change_times = [8, 10, 12]
            self.movement_threshold = 0.01 # Lower value means more sensitive
            self.penalty_score = 1
        elif difficulty == 'medium':
            self.light_change_times = [5, 7, 9]
            self.movement_threshold = 0.005
            self.penalty_score = 2
        else:  # hard
            self.light_change_times = [3, 4, 5]
            self.movement_threshold = 0.001
            self.penalty_score = 3
        
        self.cur_light_duration = random.choice(self.light_change_times)
        self.last_change_time = time.time()
    
    def detect_movement(self, current_landmarks):
        """
        Detect movement during red light - calculate movement distance
        
        Args:
            current_landmarks: MediaPipe landmarks from current frame
            
        Returns:
            bool: Whether movement was detected
        """
        # Extract x, y coordinates for movement detection
        current_positions = [(lm.x, lm.y) for lm in current_landmarks]
        
        if self.prev_landmarks is None:
            self.prev_landmarks = current_positions
            return False
        
        # Compare landmarks between previous and current frame
        prev_landmarks = self.prev_landmarks
        
        # Calculate movement distance for each landmark point
        distances = []
        for i in range(min(len(prev_landmarks), len(current_positions))):
            dist = np.sqrt((prev_landmarks[i][0] - current_positions[i][0])**2 + 
                          (prev_landmarks[i][1] - current_positions[i][1])**2)
            distances.append(dist)
        
        # Calculate average movement distance
        avg_movement = np.mean(distances)
        
        # Update previous frame
        self.prev_landmarks = current_positions
        
        # Return movement detection result
        is_moving = avg_movement > self.movement_threshold
        if is_moving:
            print(f"Movement detected! Avg movement: {avg_movement:.5f}")
        
        return is_moving
    
    def interpolate_sequence(self, sequence, target_length):
        """
        Interpolate a sequence to match the required length
        
        Args:
            sequence: List of feature arrays
            target_length: Desired sequence length
            
        Returns:
            Interpolated sequence as numpy array
        """
        if len(sequence) == target_length:
            return np.array(sequence)
        
        # Create indices for the original sequence
        orig_indices = np.arange(len(sequence))
        
        # Create indices for the target sequence
        target_indices = np.linspace(0, len(sequence) - 1, target_length)
        
        # Initialize the interpolated sequence
        interpolated = []
        
        # Interpolate for each feature dimension
        for i in range(len(target_indices)):
            idx = target_indices[i]
            if idx.is_integer():
                # If index is an integer, use the exact frame
                interpolated.append(sequence[int(idx)])
            else:
                # Otherwise, linearly interpolate between adjacent frames
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))
                weight_ceil = idx - idx_floor
                weight_floor = 1 - weight_ceil
                
                frame = weight_floor * sequence[idx_floor] + weight_ceil * sequence[idx_ceil]
                interpolated.append(frame)
        
        return np.array(interpolated)
    
    def detect_action(self, features):
        """
        Use model to classify specific actions (used during green light)
        """
        if self.model is None:
            return -1, 0.0
        
        # Apply the same custom normalization as in training
        lm_array = features.reshape(-1, 4)
        normalized_features = normalize_pose(lm_array)
        
        # Add current features to sequence
        self.player_sequence.append(normalized_features)
        
        # Keep only the most recent frames
        if len(self.player_sequence) > self.buffer_size:
            self.player_sequence.pop(0)
        
        # Need a minimum number of frames to make prediction
        min_frames = 10
        if len(self.player_sequence) < min_frames:
            return -1, 0.0
        
        # If we have a partial sequence, interpolate to match buffer size
        sequence_data = self.player_sequence
        if len(sequence_data) < self.buffer_size:
            sequence_data = self.interpolate_sequence(sequence_data, self.buffer_size)
        else:
            sequence_data = np.array(sequence_data)
        
        # Make prediction
        with torch.no_grad():
            sequence = torch.FloatTensor(sequence_data).unsqueeze(0)
            outputs = self.model(sequence)
            probabilities = F.softmax(outputs, dim=1)[0]
            action_idx = torch.argmax(probabilities).item()
            confidence = probabilities[action_idx].item()
        
        # Ignore actions with low confidence
        min_confidence = 0.85 #0.4  # Lowered for more sensitivity
        if confidence < min_confidence:
            return -1, 0.0
            
        return action_idx, confidence
    
    def update_game_state(self):
        """Update game state (red/green light transitions)"""
        current_time = time.time()
        time_delta = current_time - self.last_change_time
        self.remaining_time = max(0, self.cur_light_duration - time_delta)
        self.remaining_seconds = int(self.remaining_time)
        
        # Transition between red light and green light
        if time_delta > self.cur_light_duration:
            # Changing state
            self.red_light = not self.red_light
            self.last_change_time = current_time
            self.cur_light_duration = random.choice(self.light_change_times)
            
            if self.red_light:
                # Red light started
                # Store the completed action if there is one
                if self.completed_action and self.current_required_action is not None:
                    self.last_completed_action = self.current_required_action
                
                self.current_required_action = None
                self.penalized = False
                self.prev_landmarks = None  # Reset previous frame info
                status = "Red Light!"
            else:
                # Green light started - assign a new action
                self.select_random_action()
                self.completed_action = False
                self.current_action_idx = -1  # Reset current action
                self.current_action_name = None
                self.current_confidence = 0.0
                status = "Green Light!"
            
            print(status)
    
    def select_random_action(self):
        """Randomly select an action that player needs to perform"""
        self.current_required_action = random.randint(0, len(self.class_names) - 1)
        while self.current_required_action == self.prev_action_idx:
            self.current_required_action = random.randint(0, len(self.class_names) - 1)
        self.prev_action_idx = self.current_required_action
        print(f"Required action: {self.class_names[self.current_required_action]}")
    
    def draw_player_info(self, frame, head_pos):
        """Draw player information on the frame"""
        # Calculate screen coordinates
        screen_x = int(head_pos[0] * frame.shape[1])
        screen_y = int(head_pos[1] * frame.shape[0]) - 100
        
        # Draw player info
        if self.completed_action and self.current_required_action is not None:
            # Action completed during green light
            action_text = f"Action: {self.class_names[self.current_required_action]} (COMPLETED)"
            text_color = (0, 255, 0)  # Green for completed action
        elif self.red_light and self.last_completed_action is not None:
            # Red light state but there's a previously completed action
            action_text = f"Last Action: {self.class_names[self.last_completed_action]} (COMPLETED)"
            text_color = (0, 255, 0)  # Green for completed action
        elif self.current_action_name and self.current_confidence > 0:
            # Display currently detected action
            action_text = f"Action: {self.current_action_name} ({self.current_confidence:.2f})"
            text_color = (255, 255, 255)  # White
        else:
            action_text = "No action detected"
            text_color = (255, 255, 255)  # White
            
        # Draw text with background
        text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        bg_rect = (
            max(0, screen_x - text_size[0]//2 - 10),
            max(0, screen_y - 30),
            min(text_size[0] + 20, frame.shape[1]),
            40
        )
        
        # Draw semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw player info text
        cv2.putText(frame, action_text, (bg_rect[0] + 10, screen_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Draw score info
        score_y = screen_y + 40
        score_text = f"Score: {self.player_score}"
        
        # Draw score with background
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        bg_rect = (
            max(0, screen_x - text_size[0]//2 - 10),
            max(0, score_y - 30),
            min(text_size[0] + 20, frame.shape[1]),
            40
        )
        
        # Draw semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw score text
        cv2.putText(frame, score_text, (bg_rect[0] + 10, score_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw feedback if available
        current_time = time.time()
        if self.action_result and current_time - self.feedback_timestamp < self.feedback_duration:
            result = self.action_result
            
            # Set color based on result
            if result == "PASS":
                result_color = (0, 255, 0)  # Green
            elif result.startswith("-"):
                result_color = (0, 0, 255)  # Red
            else:
                result_color = (0, 165, 255)  # Orange
            
            # Draw feedback with background
            feedback_y = score_y + 40
            text_size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            bg_rect = (
                max(0, screen_x - text_size[0]//2 - 10),
                max(0, feedback_y - 30),
                min(text_size[0] + 20, frame.shape[1]), 
                40
            )
            
            # Draw semi-transparent black background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw feedback text
            cv2.putText(frame, result, (bg_rect[0] + 10, feedback_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, result_color, 2)
        
        # Draw status (COMPLETED or PENALIZED)
        status_y = score_y + 80
        status_text = None
        status_color = None
        
        if self.red_light and self.penalized:
            status_text = "PENALIZED"
            status_color = (0, 0, 255)  # Red
        elif not self.red_light and self.completed_action:
            status_text = "COMPLETED"
            status_color = (255, 0, 0)  # Blue
            
        if status_text:
            # Draw status with background
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            bg_rect = (
                max(0, screen_x - text_size[0]//2 - 10),
                max(0, status_y - 30),
                min(text_size[0] + 20, frame.shape[1]),
                40
            )
            
            # Draw semi-transparent black background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw status text
            cv2.putText(frame, status_text, (bg_rect[0] + 10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    def draw_game_info(self, frame):
        """Draw game status information on the frame"""
        # Create a taller semi-transparent overlay for status bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display game state
        if self.red_light:
            status_color = (0, 0, 255)
            status_text = "RED LIGHT!"
        else:
            status_color = (0, 255, 0)
            status_text = "GREEN LIGHT!"
        
        # Display main status
        cv2.putText(frame, f"{status_text} ({self.remaining_seconds}s)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Display required action during green light with fixed position
        if not self.red_light and self.current_required_action is not None:
            required_text = f"Required: {self.class_names[self.current_required_action].upper()}"
            cv2.putText(frame, required_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # Display game info
        difficulty_text = f"Difficulty: {self.get_difficulty_name()} (Penalty: -{self.penalty_score})"
        cv2.putText(frame, difficulty_text, (frame.shape[1] - 400, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def run(self, webcam_index=0):
        """Run the YoungHee game using webcam input"""
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            print("Webcam not available")
            return
        
        print("=== Starting YoungHee Game ===")
        print("Press 'q' to quit, 'r' to restart")
        
        # Initialize game state
        self.last_change_time = time.time()
        self.red_light = False
        self.game_over = False
        self.player_sequence = []
        self.player_score = 0
        self.action_result = None
        self.feedback_timestamp = 0
        self.penalized = False
        self.completed_action = False
        self.prev_landmarks = None
        self.current_action_idx = -1
        self.current_action_name = None
        self.current_confidence = 0.0
        self.last_completed_action = None
        self.select_random_action()
        
        # Set higher resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Set up the pose detection
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as pose:
            while cap.isOpened() and not self.game_over:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Update game state (red/green light transitions)
                self.update_game_state()
                
                # Process frame with MediaPipe Pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Process player if pose is detected
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Calculate head position (for displaying info)
                    head_pos = (landmarks[0].x, landmarks[0].y)  # Nose landmark
                    
                    # Use different detection methods for red/green light
                    if self.red_light:
                        # Red light: Use direct movement detection
                        if not self.penalized:  # Only check if not already penalized
                            is_moving = self.detect_movement(landmarks)
                            
                            if is_moving:
                                # Apply penalty when movement is detected
                                self.player_score = max(0, self.player_score - self.penalty_score)
                                
                                # Record penalty feedback
                                self.action_result = f"-{self.penalty_score}"
                                self.feedback_timestamp = time.time()
                                self.penalized = True
                                
                                print(f"Movement detected during red light! Penalty: {self.penalty_score} points")
                        
                        # Display player info during red light
                        self.draw_player_info(frame, head_pos)
                    else:
                        # Green light: Use model-based action classification
                        if not self.completed_action:  # Skip if already completed
                            # Extract pose features
                            features = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]).flatten()
                            
                            # Detect action
                            action_idx, confidence = self.detect_action(features)
                            
                            if action_idx != -1:  # Action detected
                                # Store the detected action
                                self.current_action_idx = action_idx
                                self.current_action_name = self.class_names[action_idx]
                                self.current_confidence = confidence
                                
                                # Check if action matches required action
                                if action_idx == self.current_required_action:
                                    # Successfully performed action
                                    self.player_score += 1
                                    self.action_result = "PASS"
                                    self.feedback_timestamp = time.time()
                                    self.completed_action = True
                                    print(f"Successfully performed {self.current_action_name}! +1 point")
                                else:
                                    # Wrong action performed
                                    self.action_result = "WRONG!"
                                    self.feedback_timestamp = time.time()
                            
                            # Display player info with current action
                            self.draw_player_info(frame, head_pos)
                        else:
                            # Already completed the action
                            self.draw_player_info(frame, head_pos)
                    
                    # Determine pose color
                    pose_color = (0, 255, 0)  # Default: green
                    if self.red_light and self.penalized:
                        pose_color = (0, 0, 255)  # Red for penalized
                    elif not self.red_light and self.completed_action:
                        pose_color = (255, 0, 0)  # Blue for completed action
                    
                    # Draw pose landmarks
                    self.mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_draw.DrawingSpec(
                            color=pose_color, thickness=3, circle_radius=4
                        )
                    )
                
                # Draw game information
                self.draw_game_info(frame)
                
                # Display frame
                cv2.imshow("YoungHee Game", frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset game
                    print("=== Restarting Game ===")
                    self.last_change_time = time.time()
                    self.red_light = False
                    self.player_sequence = []
                    self.player_score = 0
                    self.action_result = None
                    self.feedback_timestamp = 0
                    self.penalized = False
                    self.completed_action = False
                    self.prev_landmarks = None
                    self.current_action_idx = -1
                    self.current_action_name = None
                    self.current_confidence = 0.0
                    self.last_completed_action = None
                    self.select_random_action()
        
        cap.release()
        cv2.destroyAllWindows()
        
    def get_difficulty_name(self):
        """Return the current difficulty level name based on penalty score"""
        if self.penalty_score == 1:
            return "Easy"
        elif self.penalty_score == 2:
            return "Medium"
        else:
            return "Hard"