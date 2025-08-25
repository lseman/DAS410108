// decision_tree.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <iostream>

namespace py = pybind11;

// =============================================================================
// Node Structure
// =============================================================================

struct Node {
    // Split information
    int feature = -1;
    double threshold = 0.0;
    
    // Children
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;
    
    // Leaf information
    double value = 0.0;                    // Regression prediction
    std::vector<int> class_counts;         // Classification counts
    
    // Diagnostics
    int n_samples = 0;
    double impurity = 0.0;
    
    bool is_leaf() const {
        return left == nullptr && right == nullptr;
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

double gini_impurity(const std::vector<int>& counts) {
    int total = 0;
    for (int count : counts) {
        total += count;
    }
    
    if (total == 0) return 0.0;
    
    double gini = 1.0;
    for (int count : counts) {
        double p = static_cast<double>(count) / total;
        gini -= p * p;
    }
    return gini;
}

double entropy(const std::vector<int>& counts) {
    int total = 0;
    for (int count : counts) {
        total += count;
    }
    
    if (total == 0) return 0.0;
    
    double ent = 0.0;
    for (int count : counts) {
        if (count > 0) {
            double p = static_cast<double>(count) / total;
            ent -= p * std::log2(p);
        }
    }
    return ent;
}

double mse_impurity(const std::vector<double>& y_values) {
    if (y_values.empty()) return 0.0;
    
    double mean = 0.0;
    for (double y : y_values) {
        mean += y;
    }
    mean /= y_values.size();
    
    double sse = 0.0;
    for (double y : y_values) {
        double diff = y - mean;
        sse += diff * diff;
    }
    return sse;
}

// =============================================================================
// Fast Split Finding (Core Algorithm)
// =============================================================================

struct SplitResult {
    int feature = -1;
    double threshold = 0.0;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> left_indices;
    std::vector<int> right_indices;
    bool valid = false;
};

// Optimized regression split finding
SplitResult find_best_split_regression(
    const py::array_t<double>& X,
    const py::array_t<double>& y,
    const std::vector<int>& sample_indices,
    int min_samples_leaf
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    
    double* X_ptr = static_cast<double*>(X_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    
    int n_samples = sample_indices.size();
    int n_features = X_buf.shape[1];
    
    SplitResult best_split;
    
    // Pre-calculate total sum and sum of squares for efficiency
    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    for (int idx : sample_indices) {
        double y_val = y_ptr[idx];
        total_sum += y_val;
        total_sum_sq += y_val * y_val;
    }
    
    // Try each feature
    for (int feature = 0; feature < n_features; ++feature) {
        // Create sorted indices for this feature
        std::vector<std::pair<double, int>> feature_values;
        feature_values.reserve(n_samples);
        
        for (int idx : sample_indices) {
            double x_val = X_ptr[idx * n_features + feature];
            feature_values.emplace_back(x_val, idx);
        }
        
        // Sort by feature value
        std::sort(feature_values.begin(), feature_values.end());
        
        // Try splits between consecutive unique values
        double left_sum = 0.0;
        double left_sum_sq = 0.0;
        
        for (int i = 0; i < n_samples - min_samples_leaf; ++i) {
            int current_idx = feature_values[i].second;
            double y_val = y_ptr[current_idx];
            
            left_sum += y_val;
            left_sum_sq += y_val * y_val;
            
            // Check if we have enough samples for a valid split
            int left_size = i + 1;
            int right_size = n_samples - left_size;
            
            if (left_size < min_samples_leaf || right_size < min_samples_leaf) {
                continue;
            }
            
            // Check if we can actually split here (different x values)
            if (i < n_samples - 1 && 
                feature_values[i].first >= feature_values[i + 1].first) {
                continue;
            }
            
            // Calculate SSE for both sides
            double left_mean = left_sum / left_size;
            double left_sse = left_sum_sq - left_size * left_mean * left_mean;
            
            double right_sum = total_sum - left_sum;
            double right_sum_sq = total_sum_sq - left_sum_sq;
            double right_mean = right_sum / right_size;
            double right_sse = right_sum_sq - right_size * right_mean * right_mean;
            
            double total_cost = left_sse + right_sse;
            
            if (total_cost < best_split.cost) {
                best_split.cost = total_cost;
                best_split.feature = feature;
                best_split.threshold = 0.5 * (feature_values[i].first + feature_values[i + 1].first);
                best_split.valid = true;
                
                // Store split indices
                best_split.left_indices.clear();
                best_split.right_indices.clear();
                
                for (int j = 0; j <= i; ++j) {
                    best_split.left_indices.push_back(feature_values[j].second);
                }
                for (int j = i + 1; j < n_samples; ++j) {
                    best_split.right_indices.push_back(feature_values[j].second);
                }
            }
        }
    }
    
    return best_split;
}

// Classification split finding
SplitResult find_best_split_classification(
    const py::array_t<double>& X,
    const py::array_t<int>& y,
    const std::vector<int>& sample_indices,
    int n_classes,
    int min_samples_leaf,
    const std::string& criterion
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    
    double* X_ptr = static_cast<double*>(X_buf.ptr);
    int* y_ptr = static_cast<int*>(y_buf.ptr);
    
    int n_samples = sample_indices.size();
    int n_features = X_buf.shape[1];
    
    SplitResult best_split;
    
    // Count total class occurrences
    std::vector<int> total_counts(n_classes, 0);
    for (int idx : sample_indices) {
        total_counts[y_ptr[idx]]++;
    }
    
    // Try each feature
    for (int feature = 0; feature < n_features; ++feature) {
        // Create sorted indices for this feature
        std::vector<std::pair<double, int>> feature_values;
        feature_values.reserve(n_samples);
        
        for (int idx : sample_indices) {
            double x_val = X_ptr[idx * n_features + feature];
            feature_values.emplace_back(x_val, idx);
        }
        
        std::sort(feature_values.begin(), feature_values.end());
        
        // Try splits
        std::vector<int> left_counts(n_classes, 0);
        
        for (int i = 0; i < n_samples - min_samples_leaf; ++i) {
            int current_idx = feature_values[i].second;
            left_counts[y_ptr[current_idx]]++;
            
            int left_size = i + 1;
            int right_size = n_samples - left_size;
            
            if (left_size < min_samples_leaf || right_size < min_samples_leaf) {
                continue;
            }
            
            if (i < n_samples - 1 && 
                feature_values[i].first >= feature_values[i + 1].first) {
                continue;
            }
            
            // Calculate right counts
            std::vector<int> right_counts(n_classes);
            for (int c = 0; c < n_classes; ++c) {
                right_counts[c] = total_counts[c] - left_counts[c];
            }
            
            // Calculate impurity
            double left_imp = (criterion == "gini") ? 
                             gini_impurity(left_counts) : entropy(left_counts);
            double right_imp = (criterion == "gini") ? 
                              gini_impurity(right_counts) : entropy(right_counts);
            
            double total_cost = left_size * left_imp + right_size * right_imp;
            
            if (total_cost < best_split.cost) {
                best_split.cost = total_cost;
                best_split.feature = feature;
                best_split.threshold = 0.5 * (feature_values[i].first + feature_values[i + 1].first);
                best_split.valid = true;
                
                // Store split indices
                best_split.left_indices.clear();
                best_split.right_indices.clear();
                
                for (int j = 0; j <= i; ++j) {
                    best_split.left_indices.push_back(feature_values[j].second);
                }
                for (int j = i + 1; j < n_samples; ++j) {
                    best_split.right_indices.push_back(feature_values[j].second);
                }
            }
        }
    }
    
    return best_split;
}

// =============================================================================
// Decision Tree Class
// =============================================================================

class FastDecisionTree {
private:
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_impurity_decrease_;
    
    int n_features_;
    int n_classes_;
    std::vector<double> classes_;
    
    std::unique_ptr<Node> root_;
    bool is_classification_;

public:
    FastDecisionTree(
        const std::string& criterion = "gini",
        int max_depth = -1,
        int min_samples_split = 2,
        int min_samples_leaf = 1,
        double min_impurity_decrease = 0.0
    ) : criterion_(criterion),
        max_depth_(max_depth > 0 ? max_depth : 1000),
        min_samples_split_(min_samples_split),
        min_samples_leaf_(min_samples_leaf),
        min_impurity_decrease_(min_impurity_decrease),
        is_classification_(criterion != "mse") {}
    
    void fit(py::array_t<double> X, py::array y_input) {
        auto X_buf = X.request();
        if (X_buf.ndim != 2) {
            throw std::runtime_error("X must be 2-dimensional");
        }
        
        n_features_ = X_buf.shape[1];
        int n_samples = X_buf.shape[0];
        
        // Handle both classification and regression targets
        std::vector<int> y_int;
        std::vector<double> y_double;
        
        if (is_classification_) {
            // Convert to integer labels and find unique classes
            auto y_buf = y_input.request();
            if (y_buf.ndim != 1) {
                throw std::runtime_error("y must be 1-dimensional");
            }
            
            std::set<int> unique_labels;
            if (y_input.dtype().is(py::dtype::of<int>())) {
                int* y_ptr = static_cast<int*>(y_buf.ptr);
                for (int i = 0; i < n_samples; ++i) {
                    y_int.push_back(y_ptr[i]);
                    unique_labels.insert(y_ptr[i]);
                }
            } else {
                // Assume double, convert to int
                double* y_ptr = static_cast<double*>(y_buf.ptr);
                for (int i = 0; i < n_samples; ++i) {
                    int label = static_cast<int>(y_ptr[i]);
                    y_int.push_back(label);
                    unique_labels.insert(label);
                }
            }
            
            classes_.assign(unique_labels.begin(), unique_labels.end());
            n_classes_ = classes_.size();
            
            // Remap labels to 0-based indices
            std::map<int, int> label_map;
            for (int i = 0; i < n_classes_; ++i) {
                label_map[classes_[i]] = i;
            }
            for (int& label : y_int) {
                label = label_map[label];
            }
            
        } else {
            // Regression
            auto y_buf = y_input.request();
            double* y_ptr = static_cast<double*>(y_buf.ptr);
            for (int i = 0; i < n_samples; ++i) {
                y_double.push_back(y_ptr[i]);
            }
        }
        
        // Create initial sample indices
        std::vector<int> sample_indices(n_samples);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        
        // Build tree
        if (is_classification_) {
            py::array_t<int> y_array = py::cast(y_int);
            root_ = build_tree_classification(X, y_array, sample_indices, 0);
        } else {
            py::array_t<double> y_array = py::cast(y_double);
            root_ = build_tree_regression(X, y_array, sample_indices, 0);
        }
    }
    
    std::vector<double> predict(py::array_t<double> X) {
        auto X_buf = X.request();
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        int n_samples = X_buf.shape[0];
        
        std::vector<double> predictions;
        predictions.reserve(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            double* sample = X_ptr + i * n_features_;
            predictions.push_back(predict_single(sample, root_.get()));
        }
        
        return predictions;
    }
    
    std::vector<std::vector<double>> predict_proba(py::array_t<double> X) {
        if (!is_classification_) {
            throw std::runtime_error("predict_proba only available for classification");
        }
        
        auto X_buf = X.request();
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        int n_samples = X_buf.shape[0];
        
        std::vector<std::vector<double>> probabilities;
        probabilities.reserve(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            double* sample = X_ptr + i * n_features_;
            Node* leaf = find_leaf(sample, root_.get());
            
            std::vector<double> probs(n_classes_, 0.0);
            int total = 0;
            for (int count : leaf->class_counts) {
                total += count;
            }
            
            if (total > 0) {
                for (int c = 0; c < n_classes_; ++c) {
                    probs[c] = static_cast<double>(leaf->class_counts[c]) / total;
                }
            } else {
                // Uniform distribution for empty leaf
                double uniform_prob = 1.0 / n_classes_;
                std::fill(probs.begin(), probs.end(), uniform_prob);
            }
            
            probabilities.push_back(probs);
        }
        
        return probabilities;
    }

private:
    std::unique_ptr<Node> build_tree_regression(
        py::array_t<double> X,
        py::array_t<double> y,
        const std::vector<int>& sample_indices,
        int depth
    ) {
        auto node = std::make_unique<Node>();
        node->n_samples = sample_indices.size();
        
        // Calculate node statistics
        std::vector<double> y_values;
        y_values.reserve(sample_indices.size());
        
        auto y_buf = y.request();
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        
        for (int idx : sample_indices) {
            y_values.push_back(y_ptr[idx]);
        }
        
        // Calculate mean and impurity
        double sum = 0.0;
        for (double val : y_values) {
            sum += val;
        }
        node->value = sum / y_values.size();
        node->impurity = mse_impurity(y_values);
        
        // Stopping conditions
        if (depth >= max_depth_ || 
            static_cast<int>(sample_indices.size()) < min_samples_split_ ||
            sample_indices.size() < 2 * min_samples_leaf_) {
            return node;
        }
        
        // Find best split
        SplitResult split = find_best_split_regression(X, y, sample_indices, min_samples_leaf_);
        
        if (!split.valid) {
            return node;
        }
        
        // Create internal node
        node->feature = split.feature;
        node->threshold = split.threshold;
        
        // Recursively build children
        node->left = build_tree_regression(X, y, split.left_indices, depth + 1);
        node->right = build_tree_regression(X, y, split.right_indices, depth + 1);
        
        return node;
    }
    
    std::unique_ptr<Node> build_tree_classification(
        py::array_t<double> X,
        py::array_t<int> y,
        const std::vector<int>& sample_indices,
        int depth
    ) {
        auto node = std::make_unique<Node>();
        node->n_samples = sample_indices.size();
        node->class_counts.resize(n_classes_, 0);
        
        // Count classes
        auto y_buf = y.request();
        int* y_ptr = static_cast<int*>(y_buf.ptr);
        
        for (int idx : sample_indices) {
            node->class_counts[y_ptr[idx]]++;
        }
        
        // Calculate impurity
        node->impurity = (criterion_ == "gini") ? 
                        gini_impurity(node->class_counts) : 
                        entropy(node->class_counts);
        
        // Find majority class for leaf prediction
        auto max_it = std::max_element(node->class_counts.begin(), node->class_counts.end());
        node->value = std::distance(node->class_counts.begin(), max_it);
        
        // Stopping conditions
        if (depth >= max_depth_ || 
            static_cast<int>(sample_indices.size()) < min_samples_split_ ||
            sample_indices.size() < 2 * min_samples_leaf_) {
            return node;
        }
        
        // Find best split
        SplitResult split = find_best_split_classification(X, y, sample_indices, n_classes_, 
                                                         min_samples_leaf_, criterion_);
        
        if (!split.valid) {
            return node;
        }
        
        // Create internal node
        node->feature = split.feature;
        node->threshold = split.threshold;
        
        // Recursively build children
        node->left = build_tree_classification(X, y, split.left_indices, depth + 1);
        node->right = build_tree_classification(X, y, split.right_indices, depth + 1);
        
        return node;
    }
    
    double predict_single(double* sample, Node* node) {
        while (!node->is_leaf()) {
            if (sample[node->feature] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        
        if (is_classification_) {
            return classes_[static_cast<int>(node->value)];
        } else {
            return node->value;
        }
    }
    
    Node* find_leaf(double* sample, Node* node) {
        while (!node->is_leaf()) {
            if (sample[node->feature] <= node->threshold) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        return node;
    }
};

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(fast_tree, m) {
    m.doc() = "Fast C++ Decision Tree Implementation";
    
    py::class_<FastDecisionTree>(m, "FastDecisionTree")
        .def(py::init<const std::string&, int, int, int, double>(),
             py::arg("criterion") = "gini",
             py::arg("max_depth") = -1,
             py::arg("min_samples_split") = 2,
             py::arg("min_samples_leaf") = 1,
             py::arg("min_impurity_decrease") = 0.0)
        .def("fit", &FastDecisionTree::fit)
        .def("predict", &FastDecisionTree::predict)
        .def("predict_proba", &FastDecisionTree::predict_proba);
    
    // Utility functions for testing
    m.def("gini_impurity", &gini_impurity, "Calculate Gini impurity");
    m.def("entropy", &entropy, "Calculate entropy");
    m.def("mse_impurity", &mse_impurity, "Calculate MSE impurity");
}