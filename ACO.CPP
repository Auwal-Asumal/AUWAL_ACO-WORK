#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

// Structure to represent a point
struct Point2D {
    double x, y;
    Point2D(double x_ = 0, double y_ = 0) : x(x_), y(y_) {}
};

// Class to represent an ellipse
class Ellipse {
public:
    Point2D center;
    double a;      // semi-major axis
    double b;      // semi-minor axis
    double theta;  // rotation angle in radians
    
    // Constructor
    Ellipse(Point2D center_ = Point2D(), double a_ = 0, double b_ = 0, double theta_ = 0)
        : center(center_), a(a_), b(b_), theta(theta_) {}
    
    // Generate points on the ellipse for visualization
    vector<Point2D> getPoints(int n = 100) const {
        vector<Point2D> points;
        for (int i = 0; i < n; i++) {
            double t = 2 * M_PI * i / n;
            double x = center.x + a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
            double y = center.y + a * cos(t) * sin(theta) + b * sin(t) * cos(theta);
            points.push_back(Point2D(x, y));
        }
        return points;
    }
    
    // Calculate distance from a point to ellipse
    double pointDistance(const Point2D& point) const {
        double x_trans = point.x - center.x;
        double y_trans = point.y - center.y;
        
        // Rotate point to align with ellipse axes
        double x_rot = x_trans * cos(theta) + y_trans * sin(theta);
        double y_rot = -x_trans * sin(theta) + y_trans * cos(theta);
        
        // Calculate algebraic distance
        return abs((x_rot * x_rot) / (a * a) + (y_rot * y_rot) / (b * b) - 1);
    }
    
    // Calculate fitness based on how many edge points lie on the ellipse
    double fitness(const vector<Point2D>& edge_points, double threshold = 2.0) const {
        if (edge_points.empty()) return 0;
        
        int on_ellipse = 0;
        for (const auto& point : edge_points) {
            if (pointDistance(point) < threshold) {
                on_ellipse++;
            }
        }
        
        // Normalize by total points and ellipse size
        double coverage = static_cast<double>(on_ellipse) / edge_points.size();
        double size_penalty = 1.0 / (1.0 + a + b);  // Penalize very large ellipses
        
        return coverage * size_penalty;
    }
};

// Main ACO Ellipse Detection class
class ACOEllipseDetection {
private:
    // Parameters
    int n_ants;
    int n_iterations;
    double pheromone_decay;
    double alpha;
    double beta;
    int n_best;
    int min_edge_points;
    double ellipse_threshold;
    
    // Edge points and pheromone
    vector<Point2D> edge_points;
    vector<double> pheromone;
    
    // Best ellipse found
    shared_ptr<Ellipse> best_ellipse;
    double best_fitness;
    
    // Random number generator
    mt19937 rng;
    uniform_real_distribution<double> dist;
    
    // Fitness history for visualization
    vector<double> fitness_history;
    
public:
    // Constructor
    ACOEllipseDetection(int n_ants_ = 20, int n_iterations_ = 100, 
                       double pheromone_decay_ = 0.1, double alpha_ = 1.0,
                       double beta_ = 2.0, int n_best_ = 5, 
                       int min_edge_points_ = 10, double ellipse_threshold_ = 2.0)
        : n_ants(n_ants_), n_iterations(n_iterations_), 
          pheromone_decay(pheromone_decay_), alpha(alpha_), beta(beta_),
          n_best(n_best_), min_edge_points(min_edge_points_),
          ellipse_threshold(ellipse_threshold_), best_fitness(0),
          rng(random_device{}()), dist(0.0, 1.0) {}
    
    // Preprocess image to extract edge points
    vector<Point2D> preprocessImage(const Mat& image, 
                                    int canny_low = 50, 
                                    int canny_high = 150) {
        Mat gray, blurred, edges;
        
        // Convert to grayscale if needed
        if (image.channels() == 3) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // Apply Gaussian blur
        GaussianBlur(gray, blurred, Size(5, 5), 1.5);
        
        // Apply Canny edge detection
        Canny(blurred, edges, canny_low, canny_high);
        
        // Get edge points coordinates
        vector<Point2D> points;
        for (int y = 0; y < edges.rows; y++) {
            for (int x = 0; x < edges.cols; x++) {
                if (edges.at<uchar>(y, x) > 0) {
                    points.push_back(Point2D(x, y));
                }
            }
        }
        
        cout << "Found " << points.size() << " edge points" << endl;
        return points;
    }
    
    // Initialize pheromone matrix
    void initializePheromone(size_t n_points) {
        pheromone.resize(n_points);
        fill(pheromone.begin(), pheromone.end(), 0.1);
    }
    
    // Select edge points for an ant using probabilistic selection
    vector<Point2D> selectEdgePoints(int ant_idx, int n_points_needed = 5) {
        vector<Point2D> selected_points;
        vector<int> available_indices(edge_points.size());
        iota(available_indices.begin(), available_indices.end(), 0);
        
        for (int i = 0; i < n_points_needed && !available_indices.empty(); i++) {
            // Calculate probabilities
            vector<double> probabilities;
            double sum = 0.0;
            
            for (int idx : available_indices) {
                double prob = pow(pheromone[idx], alpha);
                probabilities.push_back(prob);
                sum += prob;
            }
            
            // Normalize
            if (sum > 0) {
                for (double& prob : probabilities) {
                    prob /= sum;
                }
            } else {
                // Equal probabilities if all are zero
                fill(probabilities.begin(), probabilities.end(), 
                     1.0 / available_indices.size());
            }
            
            // Roulette wheel selection
            double r = dist(rng);
            double cumulative = 0.0;
            int selected_idx = -1;
            
            for (size_t j = 0; j < probabilities.size(); j++) {
                cumulative += probabilities[j];
                if (r <= cumulative) {
                    selected_idx = available_indices[j];
                    break;
                }
            }
            
            if (selected_idx == -1) {
                selected_idx = available_indices[0];
            }
            
            // Add selected point
            selected_points.push_back(edge_points[selected_idx]);
            
            // Remove selected index
            auto it = find(available_indices.begin(), available_indices.end(), selected_idx);
            if (it != available_indices.end()) {
                available_indices.erase(it);
            }
        }
        
        return selected_points;
    }
    
    // Fit ellipse to given points using direct least squares
    shared_ptr<Ellipse> fitEllipseFromPoints(const vector<Point2D>& points) {
        if (points.size() < 5) {
            return nullptr;
        }
        
        try {
            // Convert points to Eigen matrices
            int n = points.size();
            MatrixXd D(n, 6);
            
            // Calculate centroid
            double sum_x = 0, sum_y = 0;
            for (const auto& p : points) {
                sum_x += p.x;
                sum_y += p.y;
            }
            Point2D centroid(sum_x / n, sum_y / n);
            
            // Build design matrix
            for (int i = 0; i < n; i++) {
                double x = points[i].x - centroid.x;
                double y = points[i].y - centroid.y;
                
                D(i, 0) = x * x;
                D(i, 1) = x * y;
                D(i, 2) = y * y;
                D(i, 3) = x;
                D(i, 4) = y;
                D(i, 5) = 1.0;
            }
            
            // Build scatter matrix
            MatrixXd S = D.transpose() * D;
            
            // Build constraint matrix for ellipse
            MatrixXd C = MatrixXd::Zero(6, 6);
            C(0, 2) = 2.0;
            C(1, 1) = -1.0;
            C(2, 0) = 2.0;
            
            // Solve generalized eigenvalue problem
            GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(S, C);
            
            if (solver.info() != Success) {
                return simpleEllipseFitting(points);
            }
            
            VectorXd eigenvalues = solver.eigenvalues().real();
            MatrixXd eigenvectors = solver.eigenvectors().real();
            
            // Find eigenvector with positive eigenvalues
            vector<int> pos_indices;
            for (int i = 0; i < eigenvalues.size(); i++) {
                if (eigenvalues(i) > 0) {
                    pos_indices.push_back(i);
                }
            }
            
            if (pos_indices.empty()) {
                return nullptr;
            }
            
            // Get the eigenvector corresponding to the smallest positive eigenvalue
            int best_idx = pos_indices[0];
            double min_eigenvalue = eigenvalues(best_idx);
            for (int idx : pos_indices) {
                if (eigenvalues(idx) < min_eigenvalue) {
                    min_eigenvalue = eigenvalues(idx);
                    best_idx = idx;
                }
            }
            
            VectorXd params = eigenvectors.col(best_idx);
            
            // Extract ellipse parameters
            double A = params(0), B = params(1), C_val = params(2);
            double D_val = params(3), E = params(4), F = params(5);
            
            // Convert to canonical form
            double denominator = B * B - 4 * A * C_val;
            if (fabs(denominator) < 1e-10) {
                return nullptr;
            }
            
            // Calculate center
            double center_x = (2 * C_val * D_val - B * E) / denominator;
            double center_y = (2 * A * E - B * D_val) / denominator;
            
            Point2D center(center_x + centroid.x, center_y + centroid.y);
            
            // Calculate major/minor axes and rotation
            double numerator = 2 * (A * E * E + C_val * D_val * D_val + 
                                   F * B * B - 2 * B * D_val * E - A * C_val * F);
            double term = sqrt((A - C_val) * (A - C_val) + B * B);
            double denominator1 = (B * B - 4 * A * C_val) * (term - (A + C_val));
            double denominator2 = (B * B - 4 * A * C_val) * (-term - (A + C_val));
            
            if (denominator1 <= 0 || denominator2 <= 0) {
                return nullptr;
            }
            
            double a = sqrt(numerator / denominator1);
            double b = sqrt(numerator / denominator2);
            
            double theta;
            if (a < b) {
                swap(a, b);
                theta = 0.5 * atan2(B, A - C_val) + M_PI / 2;
            } else {
                theta = 0.5 * atan2(B, A - C_val);
            }
            
            // Normalize angle to [0, pi]
            theta = fmod(theta, M_PI);
            if (theta < 0) theta += M_PI;
            
            // Reject invalid ellipses
            if (a < 5 || b < 5 || a > 1000 || b > 1000) {
                return nullptr;
            }
            
            return make_shared<Ellipse>(center, a, b, theta);
            
        } catch (const exception& e) {
            cerr << "Ellipse fitting error: " << e.what() << endl;
            return simpleEllipseFitting(points);
        }
    }
    
    // Simple ellipse fitting using moment-based method
    shared_ptr<Ellipse> simpleEllipseFitting(const vector<Point2D>& points) {
        if (points.size() < 5) {
            return nullptr;
        }
        
        // Calculate centroid
        double sum_x = 0, sum_y = 0;
        for (const auto& p : points) {
            sum_x += p.x;
            sum_y += p.y;
        }
        Point2D centroid(sum_x / points.size(), sum_y / points.size());
        
        // Center points and calculate second moments
        double xx = 0, yy = 0, xy = 0;
        for (const auto& p : points) {
            double dx = p.x - centroid.x;
            double dy = p.y - centroid.y;
            xx += dx * dx;
            yy += dy * dy;
            xy += dx * dy;
        }
        xx /= points.size();
        yy /= points.size();
        xy /= points.size();
        
        // Calculate orientation
        double theta = 0.5 * atan2(2 * xy, xx - yy);
        
        // Rotate points to align with axes
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        
        double x_rot_sq_sum = 0, y_rot_sq_sum = 0;
        for (const auto& p : points) {
            double dx = p.x - centroid.x;
            double dy = p.y - centroid.y;
            double x_rot = dx * cos_theta + dy * sin_theta;
            double y_rot = -dx * sin_theta + dy * cos_theta;
            x_rot_sq_sum += x_rot * x_rot;
            y_rot_sq_sum += y_rot * y_rot;
        }
        
        // Calculate axes lengths
        double a = 2 * sqrt(x_rot_sq_sum / points.size());
        double b = 2 * sqrt(y_rot_sq_sum / points.size());
        
        // Ensure a is major axis
        if (a < b) {
            swap(a, b);
            theta += M_PI / 2;
        }
        
        // Normalize angle
        theta = fmod(theta, M_PI);
        if (theta < 0) theta += M_PI;
        
        return make_shared<Ellipse>(centroid, a, b, theta);
    }
    
    // Update pheromone levels based on ellipse fitness
    void updatePheromone(const vector<shared_ptr<Ellipse>>& ellipses, 
                        const vector<double>& fitness_values) {
        // Evaporate pheromone
        for (double& p : pheromone) {
            p *= (1.0 - pheromone_decay);
        }
        
        // Create indices and sort by fitness
        vector<int> indices(ellipses.size());
        iota(indices.begin(), indices.end(), 0);
        
        sort(indices.begin(), indices.end(), 
             [&](int i, int j) { return fitness_values[i] > fitness_values[j]; });
        
        // Deposit pheromone from best ants
        for (int i = 0; i < min(n_best, (int)indices.size()); i++) {
            int idx = indices[i];
            if (fitness_values[idx] > 0) {
                double deposit = fitness_values[idx] * 10;
                auto ellipse = ellipses[idx];
                
                // Find edge points close to this ellipse
                for (size_t j = 0; j < edge_points.size(); j++) {
                    double dist = ellipse->pointDistance(edge_points[j]);
                    if (dist < ellipse_threshold) {
                        pheromone[j] += deposit / (1.0 + dist);
                    }
                }
            }
        }
    }
    
    // Remove edge points that belong to detected ellipse
    void removeDetectedPoints(const shared_ptr<Ellipse>& ellipse) {
        vector<Point2D> new_edge_points;
        vector<double> new_pheromone;
        
        int removed_count = 0;
        for (size_t i = 0; i < edge_points.size(); i++) {
            if (ellipse->pointDistance(edge_points[i]) >= ellipse_threshold) {
                new_edge_points.push_back(edge_points[i]);
                new_pheromone.push_back(pheromone[i]);
            } else {
                removed_count++;
            }
        }
        
        edge_points = move(new_edge_points);
        pheromone = move(new_pheromone);
        
        cout << "Removed " << removed_count << " points belonging to detected ellipse" << endl;
    }
    
    // Main detection function
    vector<shared_ptr<Ellipse>> detect(const Mat& image, int n_ellipses = 3) {
        // Preprocess image
        edge_points = preprocessImage(image);
        
        if (edge_points.size() < min_edge_points) {
            cout << "Not enough edge points found!" << endl;
            return {};
        }
        
        // Initialize pheromone
        initializePheromone(edge_points.size());
        
        vector<shared_ptr<Ellipse>> detected_ellipses;
        
        // Detect multiple ellipses
        for (int ellipse_num = 0; ellipse_num < n_ellipses; ellipse_num++) {
            cout << "\nDetecting ellipse " << ellipse_num + 1 << "/" << n_ellipses << endl;
            
            best_ellipse = nullptr;
            best_fitness = 0;
            fitness_history.clear();
            
            // ACO main loop
            for (int iteration = 0; iteration < n_iterations; iteration++) {
                vector<shared_ptr<Ellipse>> ellipses;
                vector<double> fitness_values;
                
                // Each ant constructs a solution
                for (int ant = 0; ant < n_ants; ant++) {
                    // Select edge points for this ant
                    auto selected_points = selectEdgePoints(ant, 10);
                    
                    // Fit ellipse to selected points
                    auto ellipse = fitEllipseFromPoints(selected_points);
                    
                    if (ellipse) {
                        // Calculate fitness
                        double fitness = ellipse->fitness(edge_points, ellipse_threshold);
                        
                        ellipses.push_back(ellipse);
                        fitness_values.push_back(fitness);
                        
                        // Update best ellipse
                        if (fitness > best_fitness) {
                            best_fitness = fitness;
                            best_ellipse = ellipse;
                        }
                    } else {
                        ellipses.push_back(nullptr);
                        fitness_values.push_back(0);
                    }
                }
                
                // Filter valid ellipses
                vector<shared_ptr<Ellipse>> valid_ellipses;
                vector<double> valid_fitness;
                
                for (size_t i = 0; i < ellipses.size(); i++) {
                    if (ellipses[i]) {
                        valid_ellipses.push_back(ellipses[i]);
                        valid_fitness.push_back(fitness_values[i]);
                    }
                }
                
                // Update pheromone
                if (!valid_ellipses.empty()) {
                    updatePheromone(valid_ellipses, valid_fitness);
                }
                
                // Track fitness history
                fitness_history.push_back(best_fitness);
                
                if (iteration % 20 == 0) {
                    cout << "Iteration " << iteration << ": Best Fitness = " 
                         << best_fitness << endl;
                }
            }
            
            if (best_ellipse && best_fitness > 0.1) {
                detected_ellipses.push_back(best_ellipse);
                removeDetectedPoints(best_ellipse);
            } else {
                break;
            }
        }
        
        return detected_ellipses;
    }
    
    // Visualize detected ellipses
    void visualizeResults(const Mat& original_image, 
                         const vector<shared_ptr<Ellipse>>& ellipses,
                         const string& window_name = "Detected Ellipses") {
        Mat display_image;
        if (original_image.channels() == 1) {
            cvtColor(original_image, display_image, COLOR_GRAY2BGR);
        } else {
            display_image = original_image.clone();
        }
        
        // Colors for different ellipses
        vector<Scalar> colors = {
            Scalar(0, 255, 0),    // Green
            Scalar(255, 0, 0),    // Blue
            Scalar(0, 0, 255),    // Red
            Scalar(255, 255, 0),  // Cyan
            Scalar(255, 0, 255)   // Magenta
        };
        
        // Draw detected ellipses
        for (size_t i = 0; i < ellipses.size(); i++) {
            Scalar color = colors[i % colors.size()];
            auto ellipse = ellipses[i];
            
            Point center(static_cast<int>(ellipse->center.x), 
                        static_cast<int>(ellipse->center.y));
            Size axes(static_cast<int>(ellipse->a), 
                     static_cast<int>(ellipse->b));
            double angle = ellipse->theta * 180.0 / M_PI;
            
            // Draw ellipse
            cv::ellipse(display_image, center, axes, angle, 0, 360, color, 2);
            
            // Draw center
            circle(display_image, center, 5, color, -1);
            
            // Add label
            string label = "Ellipse " + to_string(i + 1);
            putText(display_image, label, Point(center.x + 10, center.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        
        // Display results
        namedWindow(window_name, WINDOW_AUTOSIZE);
        imshow(window_name, display_image);
        
        // Create a separate window for edge points
        if (!edge_points.empty()) {
            Mat edge_display = Mat::zeros(original_image.size(), CV_8UC3);
            for (const auto& point : edge_points) {
                circle(edge_display, Point(static_cast<int>(point.x), 
                      static_cast<int>(point.y)), 1, Scalar(0, 0, 255), -1);
            }
            namedWindow("Edge Points", WINDOW_AUTOSIZE);
            imshow("Edge Points", edge_display);
        }
        
        // Display convergence history
        if (!fitness_history.empty()) {
            int width = 600, height = 400;
            Mat convergence = Mat::zeros(height, width, CV_8UC3);
            
            // Find min and max for scaling
            double max_fitness = *max_element(fitness_history.begin(), fitness_history.end());
            if (max_fitness > 0) {
                // Draw axes
                line(convergence, Point(50, 50), Point(50, height - 50), Scalar(255, 255, 255), 2);
                line(convergence, Point(50, height - 50), Point(width - 50, height - 50), Scalar(255, 255, 255), 2);
                
                // Draw fitness curve
                for (size_t i = 1; i < fitness_history.size(); i++) {
                    Point p1(50 + static_cast<int>((i - 1) * (width - 100) / fitness_history.size()),
                            height - 50 - static_cast<int>(fitness_history[i - 1] * (height - 100) / max_fitness));
                    Point p2(50 + static_cast<int>(i * (width - 100) / fitness_history.size()),
                            height - 50 - static_cast<int>(fitness_history[i] * (height - 100) / max_fitness));
                    line(convergence, p1, p2, Scalar(0, 255, 255), 2);
                }
                
                // Add labels
                putText(convergence, "Convergence History", Point(width/2 - 100, 30),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
                putText(convergence, "Iteration", Point(width/2 - 50, height - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                putText(convergence, "Fitness", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
                
                namedWindow("Convergence History", WINDOW_AUTOSIZE);
                imshow("Convergence History", convergence);
            }
        }
        
        waitKey(0);
    }
    
    // Getter for fitness history
    const vector<double>& getFitnessHistory() const { return fitness_history; }
    
    // Getter for edge points
    const vector<Point2D>& getEdgePoints() const { return edge_points; }
};

// Function to generate a test image with ellipses
Mat generateTestImage(Size size = Size(400, 400), int n_ellipses = 2) {
    Mat image = Mat::zeros(size, CV_8UC1);
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist_center_x(100, size.width - 100);
    uniform_int_distribution<int> dist_center_y(100, size.height - 100);
    uniform_int_distribution<int> dist_a(40, 100);
    uniform_int_distribution<int> dist_b(20, 80);
    uniform_int_distribution<int> dist_angle(0, 180);
    
    for (int i = 0; i < n_ellipses; i++) {
        Point center(dist_center_x(rng), dist_center_y(rng));
        Size axes(dist_a(rng), dist_b(rng));
        int angle = dist_angle(rng);
        
        cv::ellipse(image, center, axes, angle, 0, 360, Scalar(255), 2);
    }
    
    // Add some noise
    Mat noise = Mat(size, CV_8UC1);
    randn(noise, 0, 30);
    image = image + noise;
    
    // Threshold
    threshold(image, image, 127, 255, THRESH_BINARY);
    
    return image;
}

// Main function
int main() {
    cout << "=== ACO Ellipse Detection in C++ ===" << endl;
    
    // Generate synthetic test image
    cout << "\nGenerating synthetic test image..." << endl;
    Mat test_image = generateTestImage(Size(400, 400), 2);
    
    // Or load an image from file
    // Mat test_image = imread("path/to/your/image.jpg", IMREAD_GRAYSCALE);
    
    // Initialize ACO Ellipse Detector
    ACOEllipseDetection detector(
        15,    // n_ants
        80,    // n_iterations
        0.1,   // pheromone_decay
        1.0,   // alpha
        2.0,   // beta
        3,     // n_best
        20,    // min_edge_points
        2.0    // ellipse_threshold
    );
    
    // Detect ellipses
    cout << "\nStarting ellipse detection with ACO..." << endl;
    auto detected_ellipses = detector.detect(test_image, 2);
    
    // Display results
    cout << "\nDetected " << detected_ellipses.size() << " ellipses:" << endl;
    for (size_t i = 0; i < detected_ellipses.size(); i++) {
        auto ellipse = detected_ellipses[i];
        cout << "Ellipse " << i + 1 << ":" << endl;
        cout << "  Center: (" << ellipse->center.x << ", " << ellipse->center.y << ")" << endl;
        cout << "  Axes: " << ellipse->a << " x " << ellipse->b << endl;
        cout << "  Angle: " << ellipse->theta * 180.0 / M_PI << "°" << endl;
        cout << "  Fitness: " << ellipse->fitness(detector.getEdgePoints()) << endl;
    }
    
    // Visualize results
    detector.visualizeResults(test_image, detected_ellipses);
    
    // Save results to file
    imwrite("detected_ellipses.jpg", test_image);
    cout << "\nResults saved to 'detected_ellipses.jpg'" << endl;
    
    return 0;
}
