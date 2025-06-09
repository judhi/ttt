import cv2
import numpy as np
import pytesseract
from collections import deque
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SlidingPuzzleSolver:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.current_state = None
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # 0 represents empty space
        
    def rotate_frame_180(self, frame):
        """Rotate frame 180 degrees"""
        return cv2.rotate(frame, cv2.ROTATE_180)
        
    def detect_red_border(self, frame):
        """Detect the red border of the puzzle board"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for red color
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the puzzle board)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding and make it square
            padding = 10
            size = max(w, h) - 2 * padding
            center_x = x + w // 2
            center_y = y + h // 2
            
            x_start = center_x - size // 2
            y_start = center_y - size // 2
            
            return (x_start, y_start, size, size)
        
        return None
    
    def extract_puzzle_grid(self, frame, board_rect):
        """Extract the 3x3 grid from the detected board"""
        x, y, w, h = board_rect
        board = frame[y:y+h, x:x+w]
        
        # Divide into 3x3 grid
        cell_height = h // 3
        cell_width = w // 3
        
        grid = []
        for i in range(3):
            row = []
            for j in range(3):
                cell_y = i * cell_height
                cell_x = j * cell_width
                cell = board[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
                row.append(cell)
            grid.append(row)
        
        return grid
    
    def recognize_number(self, cell_image):
        """Recognize the number in a cell using OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        
        # Save cell image for debugging
        height, width = gray.shape
        print(f"Cell dimensions: {width}x{height}")
        
        # Calculate statistics
        variance = np.var(gray)
        mean_val = np.mean(gray)
        
        print(f"Variance: {variance:.1f}, Mean: {mean_val:.1f}")
        
        # NEW LOGIC: Use variance to distinguish empty cells from numbered cells
        # Empty cells should have very low variance (uniform color)
        # Numbered cells should have high variance (text creates contrast)
        
        if variance < 500:  # Low variance = likely empty space
            print("Detected as empty space (low variance)")
            return 0
        
        print("Detected as numbered cell (high variance), attempting OCR...")
        
        # Enhanced preprocessing for better OCR
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Try multiple thresholding methods
        # Method 1: Simple threshold
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Otsu's threshold
        _, thresh3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresholded_images = [thresh1, thresh2, thresh3]
        method_names = ["Simple", "Adaptive", "Otsu"]
        
        for i, (thresh, method_name) in enumerate(zip(thresholded_images, method_names)):
            # Invert if background is dark
            if np.mean(thresh) < 127:
                thresh = cv2.bitwise_not(thresh)
            
            # Resize for better OCR
            resized = cv2.resize(thresh, (200, 200))  # Even larger for better OCR
            
            # Apply morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            try:
                # Multiple OCR attempts with different configurations
                configs = [
                    '--psm 10 -c tessedit_char_whitelist=12345678',
                    '--psm 8 -c tessedit_char_whitelist=12345678',
                    '--psm 7 -c tessedit_char_whitelist=12345678',
                    '--psm 6 -c tessedit_char_whitelist=12345678',
                    '--psm 13 -c tessedit_char_whitelist=12345678'
                ]
                
                for config in configs:
                    text = pytesseract.image_to_string(cleaned, config=config).strip()
                    
                    # Clean up the text (remove any non-digit characters)
                    cleaned_text = ''.join(filter(str.isdigit, text))
                    
                    print(f"OCR result ({method_name}): '{text}' -> '{cleaned_text}'")
                    
                    if cleaned_text and len(cleaned_text) == 1 and 1 <= int(cleaned_text) <= 8:
                        print(f"Successfully recognized: {cleaned_text}")
                        return int(cleaned_text)
                
            except Exception as e:
                print(f"OCR error with {method_name}: {e}")
                continue
        
        # If all OCR methods fail, try simple pattern matching
        print("All OCR methods failed, trying pattern matching...")
        return self.recognize_by_pattern_matching(gray)
    
    def recognize_by_pattern_matching(self, gray_cell):
        """Simple pattern matching as backup"""
        # Check if it's likely an empty cell based on uniformity
        variance = np.var(gray_cell)
        mean_val = np.mean(gray_cell)
        
        print(f"Pattern matching - Variance: {variance:.1f}, Mean: {mean_val:.1f}")
        
        if variance < 50 and mean_val > 150:
            print("Pattern matching detected empty space")
            return 0
        
        # For numbers, we could implement template matching here
        # For now, let's try a simple approach based on image characteristics
        
        # Apply edge detection to see the shape
        edges = cv2.Canny(gray_cell, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        print(f"Edge pixels: {edge_pixels}")
        
        # Very basic heuristics (this is not robust, but might help for debugging)
        if edge_pixels < 50:
            return 0  # Probably empty
        
        # If we can't determine, return -1
        print("Could not recognize by pattern matching")
        return -1
    
    def capture_puzzle_state(self):
        """Capture and recognize the current puzzle state"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Rotate frame 180 degrees
        frame = self.rotate_frame_180(frame)
        
        # Detect red border
        board_rect = self.detect_red_border(frame)
        if board_rect is None:
            return None, frame
        
        # Extract grid
        grid_cells = self.extract_puzzle_grid(frame, board_rect)
        
        # Recognize numbers
        state = []
        for i in range(3):
            row = []
            for j in range(3):
                number = self.recognize_number(grid_cells[i][j])
                row.append(number)
            state.append(row)
        
        # Draw detection rectangle on frame for visualization
        x, y, w, h = board_rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return state, frame
    
    def is_valid_state(self, state):
        """Check if the recognized state is valid"""
        if state is None:
            print("State is None")
            return False
        
        flat_state = [num for row in state for num in row]
        expected_numbers = set(range(0, 9))
        actual_numbers = set(flat_state)
        
        print(f"Flat state: {flat_state}")
        print(f"Expected numbers: {expected_numbers}")
        print(f"Actual numbers: {actual_numbers}")
        print(f"Count of -1: {flat_state.count(-1)}")
        print(f"Sets equal: {expected_numbers == actual_numbers}")
        print(f"No unrecognized: {flat_state.count(-1) == 0}")
        
        return expected_numbers == actual_numbers and flat_state.count(-1) == 0
    
    def get_neighbors(self, state):
        """Get all possible moves from current state"""
        # Find empty space (0)
        empty_row, empty_col = None, None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    empty_row, empty_col = i, j
                    break
        
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # Create new state by swapping
                new_state = [row[:] for row in state]
                new_state[empty_row][empty_col] = new_state[new_row][new_col]
                new_state[new_row][new_col] = 0
                
                # Return the coordinates of the cell that should be clicked (1-indexed)
                cell_coordinates = (new_row + 1, new_col + 1)
                neighbors.append((new_state, cell_coordinates))
        
        return neighbors
    
    def state_to_tuple(self, state):
        """Convert state to tuple for hashing"""
        return tuple(tuple(row) for row in state)
    
    def solve_puzzle(self, start_state):
        """Solve the puzzle using BFS"""
        if start_state == self.goal_state:
            return []
        
        queue = deque([(start_state, [])])
        visited = {self.state_to_tuple(start_state)}
        
        while queue:
            current_state, moves = queue.popleft()
            
            # Get all possible moves
            neighbors = self.get_neighbors(current_state)
            
            for next_state, cell_coordinates in neighbors:
                state_tuple = self.state_to_tuple(next_state)
                
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    new_moves = moves + [cell_coordinates]
                    
                    if next_state == self.goal_state:
                        return new_moves
                    
                    queue.append((next_state, new_moves))
        
        return None  # No solution found
    
    def run(self):
        """Main loop to capture and solve puzzle"""
        print("Slider Puzzle Solver")
        print("Controls:")
        print("- Press 'c' to capture and solve the puzzle")
        print("- Press 'q' to quit")
        print("\nMake sure the puzzle board with red border is visible in the camera")
        print("Camera feed is rotated 180 degrees")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Rotate frame 180 degrees for display
            frame = self.rotate_frame_180(frame)
            
            # Show camera feed
            cv2.imshow('Puzzle Solver - Camera Feed', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                print("\nCapturing puzzle state...")
                state, detection_frame = self.capture_puzzle_state()
                
                if state is not None:
                    print("Detected state:")
                    for row in state:
                        print(row)
                    
                    if self.is_valid_state(state):
                        print("\nSolving puzzle...")
                        solution = self.solve_puzzle(state)
                        
                        if solution:
                            print(f"\nSolution found! Click these cells in order:")
                            coordinate_strings = [f"({coord[0]},{coord[1]})" for coord in solution]
                            print(f"Sequence: {' -> '.join(coordinate_strings)}")
                            print(f"Total moves: {len(solution)}")
                            print("\nCoordinate system: (row, column) where (1,1) is top-left and (3,3) is bottom-right")
                        else:
                            print("No solution found. The puzzle might be unsolvable.")
                    else:
                        print("Invalid puzzle state detected. Please ensure all numbers 1-8 are visible and the empty space is clear.")
                else:
                    print("Could not detect puzzle board. Make sure the red border is clearly visible.")
                
                # Show detection result
                if 'detection_frame' in locals():
                    cv2.imshow('Detection Result', detection_frame)
                    cv2.waitKey(2000)  # Show for 2 seconds
                    cv2.destroyWindow('Detection Result')
            
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # You might need to install pytesseract
    # pip install pytesseract
    # Also install Tesseract OCR on your system
    
    try:
        solver = SlidingPuzzleSolver(camera_index=1)
        solver.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. OpenCV installed: pip install opencv-python")
        print("2. Pytesseract installed: pip install pytesseract")
        print("3. Tesseract OCR installed on your system")
        print("4. Camera connected and accessible at index 1")

if __name__ == "__main__":
    main()
