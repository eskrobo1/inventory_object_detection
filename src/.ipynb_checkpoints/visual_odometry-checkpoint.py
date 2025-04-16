import numpy as np
import cv2 as cv
import time
from scipy.optimize import least_squares

class VisualOdometry:
    def __init__(self, camera_params_file, algorithm='orb', min_matches=10, use_ba=True):
        """
        Initialize the Visual Odometry system
        
        Args:
            camera_params_file (str): Path to camera intrinsic parameters file
            algorithm (str): Feature detection algorithm ('orb', 'sift')
            min_matches (int): Minimum number of matched points for motion estimation
            use_ba (bool): Whether to use bundle adjustment
        """
        self.min_matches = min_matches
        self.algorithm = algorithm
        self.use_ba = use_ba
        
        # Load camera intrinsic parameters
        self.K, self.dist_coeffs = self._load_camera_params(camera_params_file)
        self.f = (self.K[0, 0] + self.K[1, 1]) / 2
        #self.pp = (self.K[0, 2], self.K[1, 2])
        
        # Initialize camera position
        self.curr_R = np.eye(3)
        self.curr_t = np.zeros((3, 1))
        
        # Camera trajectory history
        self.trajectory = []
        self.trajectory.append((0, 0, 0))
        
        # Previous frame and its keypoints
        self.prev_img = None
        self.prev_kp = None
        self.prev_des = None
        
        # 3D points for scene visualization
        self.point_cloud = []
        
        # Performance measurement
        self.processing_times = []
        
    def _load_camera_params(self, camera_params_file):
        """Load camera intrinsic parameters from file"""
        try:
            K = np.loadtxt(camera_params_file)
            dist_file = camera_params_file.replace('K', 'D')
            try:
                dist_coeffs = np.loadtxt(dist_file)
            except Exception as e:
                print(f"Error loading distortion coefficients: {e}")
                print("Using zeros for distortion coefficients")
                dist_coeffs = np.zeros(5)    
            return K, dist_coeffs
        except Exception as e:
            print(f"Error reading camera parameters: {e}")
            # Default parameters if loading fails
            return np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0, 0, 1]
            ])
    
    def extract_features(self, img):
        """
        Extract feature points from an image using the selected algorithm
        
        Args:
            img: Input image
            
        Returns:
            kp: Keypoints
            des: Descriptors
            processing_time: Time taken to extract features
        """
        start_time = time.time()
        
        if self.algorithm == 'orb':
            orb = cv.ORB_create(nfeatures=3000)
            kp, des = orb.detectAndCompute(img, None)
        
        elif self.algorithm == 'sift':
            sift = cv.SIFT_create(nfeatures=3000)
            kp, des = sift.detectAndCompute(img, None)
        
        else:
            # Default: ORB
            orb = cv.ORB_create()
            kp, des = orb.detectAndCompute(img, None)
        
        processing_time = time.time() - start_time
        
        return kp, des, processing_time
    
    def match_features(self, des1, des2):
        """
        Match feature points between two frames
        
        Args:
            des1: Descriptors from the first frame
            des2: Descriptors from the second frame
            
        Returns:
            matches: List of matched point pairs
        """
        if des1 is None or des2 is None:
            return []
        
        if self.algorithm in ['orb']:
            # For binary descriptors, use Hamming distance
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
        #else:
            # TODO:For SIFT use L2 norm
        
        # Apply Lowe's ratio test to filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def estimate_motion(self, kp1, kp2, matches):
        """
        Estimate camera motion from matched feature points
        
        Args:
            kp1: Keypoints from the first frame
            kp2: Keypoints from the second frame
            matches: Matched feature points
            
        Returns:
            R: Rotation matrix
            t: Translation vector
            inliers: Inlier indicators
        """
        # Get coordinates of matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Compute essential matrix using RANSAC
        E, mask = cv.findEssentialMat(
            pts1, pts2, self.K, 
            method=cv.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        # Decompose essential matrix to get R and t
        _, R, t, mask = cv.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask
    
    def triangulate_points(self, kp1, kp2, matches, R, t):
        """
        Triangulate 3D points from keypoint pairs and estimated motion
        
        Args:
            kp1: Keypoints from the first frame
            kp2: Keypoints from the second frame
            matches: Matched feature points
            R: Rotation matrix
            t: Translation vector
            
        Returns:
            points_3d: Triangulated 3D points
        """
        # Projection matrices
        P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(self.K, np.hstack((R, t)))
        
        # Matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Check if we have any points to process
        if len(pts1) == 0 or len(pts2) == 0:
            print("Warning: No points to triangulate, returning empty point cloud")
            return np.array([])  # Return empty array
        
        # Normalize points
        pts1_norm = cv.undistortPoints(pts1.reshape(-1, 1, 2), self.K, self.dist_coeffs)
        pts2_norm = cv.undistortPoints(pts2.reshape(-1, 1, 2), self.K, self.dist_coeffs)


        # Triangulation
        points_4d = cv.triangulatePoints(
            P1, P2, 
            pts1_norm.reshape(-1, 2).T, 
            pts2_norm.reshape(-1, 2).T
        )
        
        # Convert from homogeneous coordinates
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T
    
    def bundle_adjustment(self, points_3d, points_2d, K, R_init, t_init):
        """
        Bundle adjustment for optimizing camera position and 3D points
        
        Args:
            points_3d: 3D points
            points_2d: 2D projections of points
            K: Intrinsic parameters matrix
            R_init: Initial rotation matrix
            t_init: Initial translation vector
            
        Returns:
            R_opt: Optimized rotation matrix
            t_opt: Optimized translation vector
        """
        # Convert rotation matrix to Rodrigues vector
        r_vec, _ = cv.Rodrigues(R_init)
        
        # Parameters for optimization (r_vec and t_init)
        params = np.hstack((r_vec.flatten(), t_init.flatten()))
        
        # Function to minimize (reprojection error)
        def reprojection_error(params, points_3d, points_2d, K):
            r_vec = params[:3].reshape(3, 1)
            t_vec = params[3:].reshape(3, 1)
            
            # Project 3D points onto the image
            projected_points, _ = cv.projectPoints(points_3d, r_vec, t_vec, K, None)
            projected_points = projected_points.reshape(-1, 2)
            
            # Error between projected and actual 2D points
            error = (points_2d - projected_points).ravel()
            return error
        
        # Optimization
        result = least_squares(
            reprojection_error, 
            params, 
            args=(points_3d, points_2d, K), 
            method='lm'
        )
    
        # Convert results back to R and t
        r_vec_opt = result.x[:3].reshape(3, 1)
        t_opt = result.x[3:].reshape(3, 1)
        R_opt, _ = cv.Rodrigues(r_vec_opt)
        
        return R_opt, t_opt
    
    def process_frame(self, frame):
        """
        Process a new frame for visual odometry
        
        Args:
            frame: New image/frame
            
        Returns:
            trajectory_point: New camera position (x, y, z)
            point_cloud: Current point cloud
            processing_time: Time taken to process the frame
        """
        start_time = time.time()
        
        # Convert image to grayscale if it's not already
        if len(frame.shape) == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Extract feature points
        curr_kp, curr_des, feature_time = self.extract_features(gray)
        
        # If this is the first frame, just store the points
        if self.prev_img is None:
            self.prev_img = gray
            self.prev_kp = curr_kp
            self.prev_des = curr_des
            return self.trajectory[-1], None, feature_time
        
        # Match points between current and previous frames
        matches = self.match_features(self.prev_des, curr_des)
        
        # Check if we have enough matches
        if len(matches) < self.min_matches:
            print(f"Not enough matches for motion estimation: {len(matches)} < {self.min_matches}")
            # Save current frame as previous for next iteration
            self.prev_img = gray
            self.prev_kp = curr_kp
            self.prev_des = curr_des
            return self.trajectory[-1], None, time.time() - start_time
        
        # Estimate camera motion
        R, t, mask = self.estimate_motion(self.prev_kp, curr_kp, matches)
        
        # Filter matches based on mask (inliers)
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        # Triangulate 3D points for scene visualization
        if self.prev_img is not None:
            points_3d = self.triangulate_points(self.prev_kp, curr_kp, inlier_matches, R, t)
        
        # Bundle adjustment for optimization (if enabled)
        if self.use_ba and len(inlier_matches) > 5:
            # Prepare data for bundle adjustment
            pts2d_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in inlier_matches])
            R_opt, t_opt = self.bundle_adjustment(points_3d, pts2d_prev, self.K, R, t)
            R, t = R_opt, t_opt
        
        # Update camera position
        self.curr_t = self.curr_t + self.curr_R.dot(t)
        self.curr_R = R.dot(self.curr_R)
        
        # Add new position to trajectory
        x, y, z = self.curr_t.flatten()
        self.trajectory.append((x, y, z))
        
        # Add good 3D points to point cloud (filtered)
        valid_points = []
        for point in points_3d:
            # Filter points by depth and distance
            if point[2] > 0 and np.linalg.norm(point) < 100:
                valid_points.append(point)
        
        if len(valid_points) > 0:
            self.point_cloud.extend(valid_points)
            # Limit number of points in cloud for performance
            if len(self.point_cloud) > 5000:
                self.point_cloud = self.point_cloud[-5000:]
        
        # Save current frame as previous for next iteration
        self.prev_img = gray
        self.prev_kp = curr_kp
        self.prev_des = curr_des
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return self.trajectory[-1], self.point_cloud, processing_time
    
    def get_trajectory(self):
        return self.trajectory
    
    def get_point_cloud(self):
        return self.point_cloud
    
    def get_average_processing_time(self):
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)